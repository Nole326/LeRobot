"""Privileged state control script validation."""
import json
import random
import traceback
from pathlib import Path


from runtime import prepare_args, USD_SHA256


def main():
    args, AppLauncher = prepare_args(replay=False)
    if args is None:
        return
    import numpy as np
    from common import publish, sha256
    from evaluate import check_imported_sources
    out = args.out
    import faulthandler,signal
    diagnostic_stack=(out/'python-stack.log').open('w')
    faulthandler.register(signal.SIGUSR1,file=diagnostic_stack,all_threads=False)
    assert out.is_dir() and not (out/'replay.claim').exists()
    (out/'replay.claim').touch(exist_ok=False)
    app = env = None
    try:
        publish(out/'status.json', dict(status='running', phase='resource_admission'))
        import torch
        torch.set_num_threads(2)
        assert torch.cuda.device_count() == 1, 'Expose exactly one assigned GPU'
        assert torch.cuda.mem_get_info()[0] >= args.min_free_mib * 1024**2
        assert float((torch.ones(1, device='cuda')+1).cpu()[0]) == 2
        app = AppLauncher(args).app
        import gymnasium as gym
        import leisaac.tasks
        from isaaclab.managers import RecorderTermCfg, DatasetExportMode
        from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm
        from isaaclab.utils import configclass
        from isaaclab_tasks.utils import parse_env_cfg
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
        from leisaac.utils.robot_utils import (convert_leisaac_action_to_lerobot,
                                               is_so101_at_rest_pose)
        from joint_contract import JOINTS, motor_to_rad, rad_to_motor
        from privileged_teacher import PrivilegedTeacher

        def state(e):
            term = e.cfg.terminations.success
            robot = e.scene['robot']
            q = robot.data.joint_pos[:, ids]
            plate = e.scene[term.params['plate_cfg'].name].data.root_pos_w[0]
            objects = [e.scene[o.name] for o in term.params['oranges_cfg']]
            oranges = torch.stack([o.data.root_pos_w[0] for o in objects])
            relative = oranges-plate
            bounds = relative.new_tensor([term.params.get('x_range', (-.1,.1)),
                                           term.params.get('y_range', (-.1,.1)),
                                           term.params.get('height_range', (-.07,.07))])
            inside = ((relative>bounds[:,0]) & (relative<bounds[:,1])).all(-1)
            rest = bool(is_so101_at_rest_pose(robot.data.joint_pos, robot.data.joint_names)[0])
            success = bool(term.func(e, **term.params)[0])
            assert success == (bool(inside.all()) and rest)
            return dict(joint_motor=convert_leisaac_action_to_lerobot(q)[0].tolist(),
                        joint_degrees=(q[0]*180/torch.pi).cpu().tolist(),
                        joint_velocity=robot.data.joint_vel[0,ids].cpu().tolist(),
                        plate_xyz=plate.cpu().tolist(), oranges_xyz=oranges.cpu().tolist(),
                        oranges_quat=[o.data.root_quat_w[0].cpu().tolist() for o in objects],
                        oranges_velocity=[o.data.root_lin_vel_w[0].cpu().tolist() for o in objects],
                        inside=inside.cpu().tolist(), plate_count=int(inside.sum()), at_rest=rest,
                        native_success=success)

        class Capture(RecorderTerm):
            def record_pre_reset(self, env_ids):
                if getattr(self._env, 'replay_capture_enabled', False) and bool(self._env.reset_buf.any()):
                    self._env.replay_final = state(self._env)
                return None, None

        @configclass
        class Recorder(RecorderManagerBaseCfg):
            dataset_export_mode = DatasetExportMode.EXPORT_NONE
            export_in_record_pre_reset = False
            final_state = RecorderTermCfg(class_type=Capture)

        sources = check_imported_sources()
        cfg = parse_env_cfg('LeIsaac-SO101-PickOrange-v0', device=args.device, num_envs=1)
        cfg.use_teleop_device('so101leader')
        assert sha256(Path(cfg.scene.robot.spawn.usd_path)) == USD_SHA256, 'Robot calibration identity differs'
        cfg.seed = args.seeds[0]
        cfg.episode_length_s = 55
        cfg.sim.enable_scene_query_support = True
        cfg.scene.robot.spawn.rigid_props.disable_gravity = False
        cfg.recorders = Recorder()
        assert cfg.dynamic_reset_gripper_effort_limit
        env = gym.make('LeIsaac-SO101-PickOrange-v0', cfg=cfg).unwrapped
        robot = env.scene['robot']
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS and abs(env.step_dt-1/60)<1e-7 and env.max_episode_length == 3300
        terms = [env.action_manager.get_term(k) for k in env.action_manager.active_terms]
        assert sum([list(t._joint_names) for t in terms], []) == JOINTS
        offsets = torch.cat([torch.as_tensor(t._offset, device=env.device).expand(1,t.action_dim) for t in terms], 1)
        scales = torch.cat([torch.as_tensor(t._scale, device=env.device).expand(1,t.action_dim) for t in terms], 1)
        base_out=out
        results=[]
        for scene_seed in args.seeds:
            args.seed=scene_seed
            out=base_out/f'seed-{scene_seed}'
            out.mkdir(exist_ok=False)
            random.seed(scene_seed); np.random.seed(scene_seed); torch.manual_seed(scene_seed)
            publish(out/'manifest.json', dict(protocol='privileged_teacher', seed=args.seed,
                actor_inputs=['wrist_rgb','front_rgb','joint_state'], action_hz=30, physics_hz=60,
                horizon_seconds=55, privileged_state='teacher and diagnostics only',
                sources=sources, robot_usd_sha256=USD_SHA256,
                temporary_diagnostic_images=True, no_training_dataset_yet=True))
            obs,_=env.reset(seed=args.seed)
            env.replay_capture_enabled=True; env.replay_final=None
            initial=env.scene.get_state(is_relative=True)
            torch.save(initial,out/'initial-state.pt')
            teacher=PrivilegedTeacher(cfg.scene.robot.spawn.usd_path,.2)
            teacher.diagnostic_out=out
            teacher.reset(); teacher.setup(env)
            previous=robot.data.joint_pos[:,ids].clone()
            limits=robot.data.soft_joint_pos_limits[0,ids]
            actions=[]; states=[]
            with (out/'trace.jsonl').open('w') as trace:
                for step in range(3300):
                    if step%2==0:
                        publish(out/'heartbeat.json',dict(step=step,call='teacher_command',phase=teacher.stages[teacher.stage]))
                        before=robot.data.joint_pos[:,ids].clone()
                        wanted=torch.as_tensor(teacher.command(previous[0].cpu().numpy(),before[0].cpu().numpy()),device=env.device).unsqueeze(0)
                        bound=.04 if teacher.stage==8 or (teacher.stage==7 and teacher.order_index==2) else .03
                        previous=previous+(wanted-previous).clamp(-bound,bound)
                        previous=previous.clamp(limits[:,0],limits[:,1])
                        teacher.applied_command=previous[0].cpu().numpy().copy()
                        command=(previous-offsets)/scales
                        motor=rad_to_motor(previous[0].cpu().numpy())
                        np.testing.assert_allclose(motor_to_rad(motor),previous[0].cpu().numpy(),atol=1e-5)
                        actions.append(motor); states.append(rad_to_motor(before[0].cpu().numpy()))
                    dynamic_reset_gripper_effort_limit_sim(env,'so101leader')
                    publish(out/'heartbeat.json',dict(step=step,call='env_step',phase=teacher.stages[teacher.stage]))
                    obs,_,term,trunc,_=env.step(command)
                    done=bool(term[0]|trunc[0]); current=env.replay_final if done else state(env)
                    trace.write(json.dumps(dict(step=step+1,phase=teacher.stages[teacher.stage],diagnostics=current))+'\n')
                    if (step+1)%30==0 or done:
                        publish(out/'events.json',teacher.events)
                        detail=dict(step=step+1,phase=teacher.stages[teacher.stage],orange=teacher._orange_now,
                            tcp=teacher.tcp().tolist(),object=teacher.object().tolist(),target=teacher.target.tolist(),
                            goal=teacher.goal.tolist(),command=previous[0].cpu().tolist(),
                            measured=teacher.measured().tolist(),gravity=teacher.gravity_offset.tolist(),integral=teacher.integral.tolist())
                        with (out/'control.jsonl').open('a') as f:f.write(json.dumps(detail)+'\n')
                        if (step+1)%300==0 and not done:
                            from PIL import Image
                            for camera in ('front','wrist'):
                                Image.fromarray(obs['policy'][camera][0].cpu().numpy()).save(out/f'temporary-{camera}-{step+1:04d}.png')
                        publish(out/'progress.json',dict(status='running',physics_steps=step+1,phase=teacher.stages[teacher.stage],plate_count=current['plate_count'],at_rest=current['at_rest'],teacher_geometry=teacher.pad_metrics() if not done and teacher.stage in (1,2,3) else None))
                        with (out/'scalars.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,metrics=dict(plate_count=current['plate_count'],at_rest=int(current['at_rest']),success=int(current['native_success']))))+'\n')
                    if done:break
                    publish(out/'heartbeat.json',dict(step=step,call='teacher_observe',phase=teacher.stages[teacher.stage]))
                    teacher.observe()
                    if teacher.is_episode_done:break
            np.savez(out/'numeric-actions.npz',action=np.asarray(actions),joints=np.asarray(states),timestamp=np.arange(len(actions))/30)
            result=dict(status='completed',success=current['native_success'],physics_steps=step+1,seconds=(step+1)/60,
                terminated=bool(term[0]),truncated=bool(trunc[0]),final=current,failure_reason=teacher.failure_reason,events=teacher.events,
                training_eligible=False,reason='independent action-trajectory replay verification and synchronized RGB capture required')
            publish(out/'result.json',result);publish(out/'status.json',dict(status='completed'))
            results.append(dict(seed=scene_seed,success=result['success'],seconds=result['seconds']))
            publish(base_out/'progress.json',dict(status='running',completed=len(results),results=results))
        publish(base_out/'result.json',dict(results=results,passed=all(r['success'] for r in results)))
        publish(base_out/'status.json',dict(status='completed'))
        out=base_out
    except BaseException:
        publish(out/'error.json',dict(traceback=traceback.format_exc()))
        publish(out/'status.json',dict(status='failed'));raise
    finally:
        try:
            if env is not None:env.close()
        finally:
            if app is not None:app.close()

if __name__=='__main__':main()
