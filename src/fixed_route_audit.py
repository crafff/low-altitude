"""Independent reduction of saved native traces, including full terminal ticks."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

from bluesky_diagnostic import distance_m
from fixed_route_probe import signed_geometry
from lateral_plan import local_xy
from nr_pilot import ConflictEvents, centerline_distance_m
from paper_performance import load_types
from route_completion import finite_exit_crossing


def audit(directory,scenes,cfg,deadline):
    result=json.loads((directory/'result.json').read_text())
    if not result['validation_complete']:
        raise ValueError('Only complete workloads can be fully audited')
    case_rows={f'{c["mode"]}-{c["seed"]}':c for c in result['cases']}
    metadata={}
    failures=[]
    def require(ok,detail):
        if not ok and len(failures)<100:
            failures.append(detail)
    types=load_types('configs/uav_types.json')
    expected_cases={f'{mode}-{seed}' for mode in result['modes'] for seed in result['selected_seeds']}
    require(set(case_rows)==expected_cases and len(case_rows)==len(result['cases']),'declared_case_coverage_differs')
    expected_pids={f['plan_id'] for c in result['cases'] for f in c['terminal_flights']}
    require(len(expected_pids)==sum(len(c['terminal_flights']) for c in result['cases']),'duplicate_declared_flight')
    with gzip.open(directory/'plans.jsonl.gz','rt') as handle:
        for line in handle:
            if time.perf_counter()>deadline:
                raise TimeoutError('Audit wall budget')
            plan=json.loads(line)
            pid=plan['plan_id']
            case_id=pid.rsplit('-',1)[0]
            case=case_rows[case_id]
            flight=next(f for f in case['terminal_flights'] if f['plan_id']==pid)
            require(pid==f'{case["mode"]}-{case["seed"]}-{flight["id"]}',pid+':plan_id_not_bound_to_original_flight')
            scene=scenes[case['seed']]
            route=next(c['waypoints_lat_lon_deg'] for c in scene['corridors'] if c['id']==flight['corridor_id'])
            original=next(f for f in scene['flights'] if f['id']==flight['id'])
            require(all(flight[k]==v for k,v in original.items()),pid+':original_flight_changed')
            require(plan['nominal_latlon']==route,pid+':original_geometry_changed')
            require(plan['geometry']['original_endpoint_latlon']==route[-1],pid+':endpoint_changed')
            require(plan['certificate']['conditional_certificate'],pid+':certificate_failed')
            require(all(i['conditional_capsule_check_passed'] for i in plan['certificate']['intervals']),pid+':interval_failed')
            require(len(plan['commands'])+1==len(plan['states'])==len(plan['progress']),pid+':alignment_failed')
            expected_entry=math.ceil((original['scheduled_entry_s']-1e-8)/.25)*.25
            require(flight['actual_entry_s']==expected_entry,pid+':admission_changed')
            first_final=min((k for k in range(1,len(plan['states']))
                if plan['progress'][k]['nominal_leg_index']==len(route)-2 and plan['phases'][k-1]!='turn'),default=10**9)
            require(pid not in metadata,pid+':duplicate_plan')
            metadata[pid]=dict(case_id=case_id,route=route,record=flight,first_final=first_final,
                legs=tuple(p['nominal_leg_index'] for p in plan['progress']),phases=tuple(plan['phases']),
                nominal_tas_mps=types[flight['type']]['nominal_tas_mps'])
    require(set(metadata)==expected_pids,'plan_coverage_differs_from_declared_flights')
    if set(metadata)!=expected_pids:
        return dict(source=str(directory),case_count=len(case_rows),flight_count=len(metadata),
            failures=failures,all_checks_passed=False)
    previous={}
    stats={pid:dict(samples=0,path_length_m=0.,outside_seconds=0.,max_distance_m=0.,first_exit=None,
        hold_samples=0,hold_failed=0,max_hold_course_error_deg=0.,return_seen=False,return_center_samples=0,
        return_center_failed=0) for pid in metadata}
    trackers={cid:ConflictEvents(cfg) for cid in case_rows}
    batch_time=batch_case=None
    batch={}
    ended=[]
    last_time={}
    terminal_checks=[]
    def flush():
        if batch:
            tracker=trackers[batch_case]
            tracker.observe(batch_time-.25,batch,.25)
            for acid,reason in ended:
                tracker.exit(acid,batch_time,reason)
            last_time[batch_case]=batch_time
            batch.clear()
            ended.clear()
    with gzip.open(directory/'traces.jsonl.gz','rt') as handle:
        for line in handle:
            row=json.loads(line)
            pid=row['plan_id']
            meta=metadata[pid]
            record,route=meta['record'],meta['route']
            value=row['actual']
            state=(value['lat_deg'],value['lon_deg'],row['altitude_m'],value['tas_mps'])
            if row['tick']==0:
                require(pid not in previous,pid+':duplicate_initial')
                require(list(state[:2])==route[0],pid+':initial_position_changed')
                require(row['time_s']==record['actual_entry_s'],pid+':initial_time_changed')
                previous[pid]=state
                continue
            if time.perf_counter()>deadline:
                raise TimeoutError('Audit wall budget')
            if (batch_case,batch_time)!=(meta['case_id'],row['time_s']):
                flush()
                batch_case,batch_time=meta['case_id'],row['time_s']
            before=previous[pid]
            batch[record['id']]=before
            st=stats[pid]
            st['samples']+=1
            require(row['tick']==st['samples'],pid+':missing_or_repeated_tick')
            require(row['phase']==meta['phases'][row['tick']-1],pid+':phase_differs_from_plan')
            require(row['time_s']==record['actual_entry_s']+.25*row['tick'],pid+':tick_time_changed')
            require(state[2]==106.68 and row['vs_mps']==0.,pid+':fixed_altitude_changed')
            distance=centerline_distance_m(state,route)
            st['outside_seconds']+=.25*(distance>76.2)
            st['max_distance_m']=max(st['max_distance_m'],distance)
            st['path_length_m']+=distance_m(before,state)
            p,endpoint,unit,first=signed_geometry(state[:2],route)
            b=local_xy(before[:2],route[0])
            origin_last=local_xy(route[-2],route[0])
            along=sum((b[j]-origin_last[j])*unit[j] for j in (0,1))
            course=math.degrees(math.atan2(unit[0],unit[1]))%360.
            course_error=abs((value['hdg_deg']-course+180.)%360.-180.)
            if row['tick']>=meta['first_final'] and along>=0 and course_error<=5.:
                crossing=finite_exit_crossing((*b,before[2]),(*p,state[2]),endpoint,unit,76.2,76.2,137.16)
                if crossing and crossing['within_width'] and crossing['within_height'] and st['first_exit'] is None:
                    st['first_exit']=row['tick']
            first_course=math.degrees(math.atan2(first[0],first[1]))%360.
            first_error=abs((value['hdg_deg']-first_course+180.)%360.-180.)
            cross=p[0]*first[1]-p[1]*first[0]
            if row['phase']=='lane_hold':
                st['hold_samples']+=1
                st['hold_failed']+=abs(cross-record['requested_lane_m'])>2. or first_error>5. or abs(state[3]-meta['nominal_tas_mps'])>1e-3
                st['max_hold_course_error_deg']=max(st['max_hold_course_error_deg'],first_error)
            if row['phase']=='lane_return':
                st['return_seen']=True
            if row['phase']=='center' and st['return_seen'] and meta['legs'][row['tick']]==0:
                st['return_center_samples']+=1
                st['return_center_failed']+=abs(cross)>2. or first_error>5.
            previous[pid]=state
            if row['tick']==record['actual_samples']:
                ended.append((record['id'],record['status']))
                terminal_checks.append(dict(plan_id=pid,actual_along_before_m=along,
                    course_error_deg=course_error,first_exit_tick=st['first_exit'],terminal_tick=row['tick']))
    flush()
    require(set(previous)==expected_pids,'initial_trace_coverage_differs')
    require({pid for pid,st in stats.items() if st['samples']>0}==expected_pids,'post_trace_coverage_differs')
    for cid,tracker in trackers.items():
        tracker.finish(last_time[cid])
        expected=case_rows[cid]['summary']['risk']
        require(all(v==expected[level][k] for level,values in tracker.summary().items() for k,v in values.items()),cid+':risk_reduction_differs')
    for pid,st in stats.items():
        record=metadata[pid]['record']
        require(st['samples']==record['actual_samples'],pid+':sample_total_differs')
        require(st['samples']*.25==record['flight_seconds'],pid+':terminal_interval_missing')
        require(abs(st['path_length_m']-record['path_length_m'])<1e-7,pid+':path_total_differs')
        require(st['outside_seconds']==record['outside_corridor_seconds'],pid+':containment_total_differs')
        require(abs(st['max_distance_m']-record['max_centerline_distance_m'])<1e-10,pid+':maximum_distance_differs')
        require(record['status']=='arrived' and st['first_exit']==st['samples'],pid+':first_qualified_exit_differs')
        if record['lane_request_status']=='accepted':
            require(st['hold_samples']*.25>=20 and not st['hold_failed'],pid+':lane_hold_failed')
            require(st['hold_samples']==record['lane_hold_samples'] and st['hold_failed']==record['lane_hold_failed_samples'],pid+':hold_counts_differ')
            require(st['return_center_samples']>0 and not st['return_center_failed'],pid+':return_center_failed')
            require(st['return_center_samples']==record['center_after_return_samples']
                and st['return_center_failed']==record['center_after_return_failed_samples'],pid+':return_counts_differ')
    for cid,case in case_rows.items():
        require(len(case['terminal_flights'])==len(scenes[case['seed']]['flights']),cid+':population_incomplete')
        require({f['id'] for f in case['terminal_flights']}=={f['id'] for f in scenes[case['seed']]['flights']},cid+':population_ids_differ')
    return dict(source=str(directory),case_count=len(case_rows),flight_count=len(metadata),
        aircraft_physics_samples=sum(s['samples'] for s in stats.values()),
        source_result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        failures=failures,all_checks_passed=not failures,terminal_checks=terminal_checks,
        max_hold_course_error_deg=max((s['max_hold_course_error_deg'] for s in stats.values()),default=0.))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifacts',nargs='+',required=True)
    parser.add_argument('--input-manifest',default='reports/fixed-route-20260905/audit-inputs.json')
    parser.add_argument('--wall-seconds',type=float,default=240.)
    args=parser.parse_args()
    if os.environ.get('LAB_RUN_DIR')!='/output' or Path.cwd()!=Path('/workspace'):
        raise ValueError('Use the isolated lab launcher')
    started=time.perf_counter()
    dependencies=json.loads(Path(args.input_manifest).read_text())['dependencies']
    selected={d['directory']:d for d in dependencies}
    if set(selected)!=set(args.artifacts) or len(selected)!=len(args.artifacts):
        raise ValueError('Declared read-only audit dependency selection changed')
    def verify_dependencies():
        for directory,entry in selected.items():
            for item in entry['files']:
                path=Path(directory)/item['name']
                with path.open('rb') as handle:
                    identity=hashlib.file_digest(handle,'sha256').hexdigest()
                if path.stat().st_size!=item['bytes'] or identity!=item['sha256']:
                    raise ValueError('Read-only audit input identity changed')
            declared=json.loads((Path(directory)/'result.json').read_text())
            if declared['modes']!=entry['expected_modes'] or declared['selected_seeds']!=entry['expected_seeds']:
                raise ValueError('Originally authorized case selection changed')
    verify_dependencies()
    fixture=Path('reports/policy-modes-refresh-850-20260905/scenarios.json')
    if hashlib.sha256(fixture.read_bytes()).hexdigest()!='3ed3692695fcbeb8503200e3d1b5bb1560583d9f98ac742f0727d99a65a4bf2c':
        raise ValueError('Original scenarios changed')
    scenes={s['seed']:s for s in json.loads(fixture.read_text())}
    cfg=json.loads(Path('configs/nr_pilot.json').read_text())
    rows=[]
    for directory in args.artifacts:
        row=audit(Path(directory),scenes,cfg,started+args.wall_seconds)
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('source','case_count','flight_count','all_checks_passed','failures')}),flush=True)
    verify_dependencies()
    result=dict(audits=rows,all_checks_passed=all(r['all_checks_passed'] for r in rows),wall_seconds=time.perf_counter()-started)
    Path('/output/result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['all_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
