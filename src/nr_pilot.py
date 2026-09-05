"""Small native BlueSky no-resolution population diagnostic, development seeds only."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import resource
import time

from bluesky_diagnostic import FT, distance_m, initialise, risk_flags


def arrived_on_route(route, remaining_m, radius_m):
    # A nearly closed route can finish near its origin before any turn is flown.
    return route.iactwp == len(route.wpname)-1 and remaining_m <= radius_m


def centerline_distance_m(state, waypoints):
    """Local equirectangular distance to the route polyline (diagnostic only)."""
    scale=math.pi*6371000/180
    points=[((p[1]-state[1])*scale*math.cos(math.radians(state[0])),
             (p[0]-state[0])*scale) for p in waypoints]
    distances=[]
    for a,b in zip(points,points[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        u=max(0,min(1,-(a[0]*dx+a[1]*dy)/(dx*dx+dy*dy)))
        distances.append(math.hypot(a[0]+u*dx,a[1]+u*dy))
    return min(distances)


class ConflictEvents:
    """An event is a continuous sampled violation by one unordered aircraft pair."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.active = {'lowc': {}, 'nmac': {}}
        self.exposure = {'lowc': 0.0, 'nmac': 0.0}
        self.events = []

    def observe(self, t, states, dt):
        flags = {'lowc': set(), 'nmac': set()}
        ids = sorted(states)
        for i, aid in enumerate(ids):
            for bid in ids[i+1:]:
                a, b = states[aid], states[bid]
                for level, flag in zip(('lowc', 'nmac'), risk_flags(distance_m(a,b), a[2]-b[2], self.cfg)):
                    if flag:
                        flags[level].add((aid,bid))
        for level, current in flags.items():
            active = self.active[level]
            for pair in sorted(set(active)-current):
                self._close(level, pair, t, 'separated')
            for pair in sorted(current):
                if pair not in active:
                    active[pair] = {'level':level, 'pair':list(pair), 'start_s':t, 'pair_seconds':0.0}
                active[pair]['pair_seconds'] += dt
            self.exposure[level] += len(current)*dt

    def _close(self, level, pair, t, reason):
        self.events.append(self.active[level].pop(pair) | {'end_s':t, 'end_reason':reason})

    def exit(self, acid, t, reason):
        for level, active in self.active.items():
            for pair in list(active):
                if acid in pair:
                    self._close(level, pair, t, reason)

    def finish(self, t):
        for level, active in self.active.items():
            for pair in list(active):
                self._close(level, pair, t, 'scenario_end')

    def summary(self):
        return {level: {'unordered_pair_seconds':self.exposure[level],
                        'directed_pair_seconds':2*self.exposure[level],
                        'event_count':sum(e['level']==level for e in self.events),
                        'events_ended_at_aircraft_exit':sum(e['level']==level and e['end_reason']!='separated'
                                                          for e in self.events)}
                for level in ('lowc','nmac')}


def run_scenario(bs, perf_class, types, cfg, scenario, output):
    from bluesky.core import simtime
    from bluesky.core.entity import getproxied
    from bluesky.traffic.asas import ConflictDetection, ConflictResolution
    from bluesky.tools import geo
    from bluesky.tools.aero import tas2cas

    output.mkdir()
    (output/'scenario.json').write_text(json.dumps(scenario,indent=2)+'\n')
    bs.sim.reset()
    perf_class.select()
    assert type(getproxied(bs.traf.perf)) is perf_class
    ConflictResolution.setmethod('OFF')
    ConflictDetection.setmethod('OFF')
    bs.traf.wind.clear()
    bs.traf.setnoise(False)
    simtime.setdt(cfg['dt_seconds'])
    flights = sorted(scenario['flights'], key=lambda f:(f['scheduled_entry_s'],f['id']))
    corridors = {c['id']:c['waypoints_lat_lon_deg'] for c in scenario['corridors']}
    records = {f['id']:dict(f, status='pending', actual_entry_s=None, flight_seconds=0.0,
                         path_length_m=0.0, admission_lowc_pairs=0, admission_nmac_pairs=0,
                         max_centerline_distance_m=0.0,outside_corridor_seconds=0.0)
               for f in flights}
    tracker = ConflictEvents(cfg)
    altitude = cfg['altitude_ft']*FT
    next_flight = steps = max_active = 0
    flight_seconds = max_alt_error = max_tas_error = 0.0
    horizon = max(f['scheduled_entry_s'] for f in flights) + cfg['per_flight_timeout_seconds'] + 2*cfg['dt_seconds']

    def states():
        return {acid:(float(bs.traf.lat[i]), float(bs.traf.lon[i]),
                      float(bs.traf.alt[i]), float(bs.traf.tas[i])) for i,acid in enumerate(bs.traf.id)}

    def admit(flight):
        acid, kind = flight['id'], flight['type']
        waypoints = corridors[flight['corridor_id']]
        start, first = waypoints[:2]
        heading = float(geo.qdrdist(*start,*first)[0])
        tas = types[kind]['nominal_tas_mps']
        before = states()
        assert bs.traf.cre(acid,kind,*start,heading,altitude,float(tas2cas(tas,altitude))) is True
        i = bs.traf.id2idx(acid)
        route = bs.traf.ap.route[i]
        for j,(lat,lon) in enumerate(waypoints[1:]):
            assert route.addwpt(i,f'{acid}P{j}',route.wplatlon,lat,lon,altitude,-999.)>=0
        route.direct(i,route.wpname[0])
        bs.traf.swlnav[i], bs.traf.swvnav[i], bs.traf.swvnavspd[i] = True,False,False
        records[acid].update(status='active', actual_entry_s=float(bs.sim.simt))
        for other in before.values():
            lowc,nmac = risk_flags(distance_m(start,other), altitude-other[2],cfg)
            records[acid]['admission_lowc_pairs'] += int(lowc)
            records[acid]['admission_nmac_pairs'] += int(nmac)

    start_wall = time.perf_counter()
    bs.sim.op()
    while next_flight < len(flights) or bs.traf.ntraf:
        if bs.sim.simt > horizon:
            raise RuntimeError('Global guard reached with unfinished planned population')
        while next_flight < len(flights) and flights[next_flight]['scheduled_entry_s'] <= bs.sim.simt+1e-9:
            admit(flights[next_flight])
            next_flight += 1
        before, t0 = states(), float(bs.sim.simt)
        max_active = max(max_active,len(before))
        bs.sim.step()
        t1 = float(bs.sim.simt)
        dt = t1-t0
        assert math.isclose(dt,cfg['dt_seconds'],abs_tol=1e-8), dt
        after = states()
        if set(after)!=set(before):
            raise RuntimeError('Unexpected native disappearance; missing traffic cannot be counted safe')
        tracker.observe(t0,before,dt)
        flight_seconds += len(before)*dt
        terminal = []
        for acid, state in after.items():
            if not all(math.isfinite(v) for v in state):
                raise RuntimeError('Non-finite native state')
            record = records[acid]
            record['flight_seconds'] += dt
            record['path_length_m'] += distance_m(before[acid],state)
            max_alt_error = max(max_alt_error,abs(state[2]-altitude))
            max_tas_error = max(max_tas_error,abs(state[3]-types[record['type']]['nominal_tas_mps']))
            lateral=centerline_distance_m(state,corridors[record['corridor_id']])
            record['max_centerline_distance_m']=max(record['max_centerline_distance_m'],lateral)
            if lateral>cfg['corridor_width_ft']*FT/2:
                record['outside_corridor_seconds']+=dt
            remaining = distance_m(state,corridors[record['corridor_id']][-1])
            route = bs.traf.ap.route[bs.traf.id2idx(acid)]
            arrived = arrived_on_route(route,remaining,cfg['arrival_radius_m'])
            timeout = t1-record['actual_entry_s'] >= cfg['per_flight_timeout_seconds']-1e-8
            if arrived or timeout:
                reason = 'arrived' if arrived else 'flight_timeout'
                record.update(status=reason, terminal_time_s=t1, terminal_state=state,
                              endpoint_distance_m=remaining, active_waypoint=route.iactwp)
                terminal.append((acid,reason))
        # Count exposure through the terminal interval and retain state before deletion.
        for acid, reason in terminal:
            tracker.exit(acid,t1,reason)
            bs.traf.delete(bs.traf.id2idx(acid))
        steps += 1
    tracker.finish(float(bs.sim.simt))
    wall = time.perf_counter()-start_wall
    counts = Counter(r['status'] for r in records.values())
    risks = tracker.summary()
    hours = flight_seconds/3600
    for event in tracker.events:
        a,b=(records[x] for x in event['pair'])
        event['same_corridor']=a['corridor_id']==b['corridor_id']
        event['starts_at_admission']=math.isclose(event['start_s'],max(a['actual_entry_s'],b['actual_entry_s']),abs_tol=1e-8)
    for level, value in risks.items():
        value['unordered_seconds_per_flight_hour'] = value['unordered_pair_seconds']/hours if hours else None
        value['directed_seconds_per_flight_hour'] = value['directed_pair_seconds']/hours if hours else None
        involved = {acid for e in tracker.events if e['level']==level for acid in e['pair']}
        value['aircraft_with_event'] = len(involved)
        value['zero_exposure_flight_fraction'] = (len(flights)-len(involved))/len(flights)
        value['same_corridor_event_count']=sum(e['level']==level and e['same_corridor'] for e in tracker.events)
        value['cross_corridor_event_count']=value['event_count']-value['same_corridor_event_count']
        value['events_started_at_admission']=sum(e['level']==level and e['starts_at_admission'] for e in tracker.events)
    checks = {'all_planned_terminal':sum(counts[k] for k in ('arrived','flight_timeout'))==len(flights),
              'none_missing':all(r['actual_entry_s'] is not None for r in records.values()),
              'altitude_held':max_alt_error<0.01, 'nominal_tas_held':max_tas_error<0.05,
              'no_resolution':ConflictResolution.selected() is ConflictResolution,
              'wind_off':bs.traf.wind.winddim==0,
              'admission_tick_error_bounded':all(0<=r['actual_entry_s']-r['scheduled_entry_s']<cfg['dt_seconds']+1e-8
                                                   for r in records.values()),
              'flight_denominator_reconciles':math.isclose(sum(r['flight_seconds'] for r in records.values()),flight_seconds),
              'exposure_reconciles':all(math.isclose(sum(e['pair_seconds'] for e in tracker.events if e['level']==level),
                                                  risks[level]['unordered_pair_seconds']) for level in risks)}
    result = {'seed':scenario['seed'], 'corridors':len(corridors), 'planned':len(flights),
              'completed':counts['arrived'], 'failed_timeout':counts['flight_timeout'],
              'pending_or_missing':len(flights)-counts['arrived']-counts['flight_timeout'],
              'flight_seconds':flight_seconds, 'flight_hours':hours, 'risk':risks,
              'admission_lowc_pairs':sum(r['admission_lowc_pairs'] for r in records.values()),
              'admission_nmac_pairs':sum(r['admission_nmac_pairs'] for r in records.values()),
              'type_counts':dict(Counter(f['type'] for f in flights)),
              'max_centerline_distance_m':max(r['max_centerline_distance_m'] for r in records.values()),
              'outside_corridor_aircraft_seconds':sum(r['outside_corridor_seconds'] for r in records.values()),
              'outside_corridor_flights':sum(r['outside_corridor_seconds']>0 for r in records.values()),
              'max_altitude_error_m':max_alt_error, 'max_tas_error_mps':max_tas_error,
              'max_active_aircraft':max_active, 'sim_seconds':float(bs.sim.simt), 'steps':steps,
              'wall_seconds':wall, 'steps_per_wall_second':steps/wall,
              'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'checks':checks, 'flights':list(records.values())}
    (output/'events.json').write_text(json.dumps(tracker.events,indent=2)+'\n')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    from paper_performance import load_types, install_performance
    from paper_scenarios import generate_scenario
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--types',default='configs/uav_types.json')
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    assert 0<cfg['dt_seconds']<=1 and 0<cfg['per_flight_timeout_seconds']<=1200
    assert cfg['stage']=='dev' and cfg['scenario_count']==len(cfg['seeds'])
    assert not cfg['store_full_trajectories'], 'Full trajectory output is not implemented in this pilot'
    assert cfg['earth_radius_m']==6371000.0
    assert version('bluesky-simulator')==cfg['bluesky_version']=='1.1.1'
    assert args.limit is None or args.limit>0
    seeds = cfg['seeds'] if args.limit is None else cfg['seeds'][:args.limit]
    assert seeds and len(seeds)<=16 and len(set(seeds))==len(seeds)
    types = load_types(args.types)
    output = Path(os.environ['LAB_RUN_DIR'])
    start = time.perf_counter()
    bs = initialise(Path('/tmp/bluesky-nr-pilot'))
    perf_class = install_performance(bs,types)
    init_seconds = time.perf_counter()-start
    results=[]
    for seed in seeds:
        scenario = generate_scenario(cfg,seed,list(types))
        result = run_scenario(bs,perf_class,types,cfg,scenario,output/f'seed-{seed}')
        results.append(result)
        print(json.dumps({k:result[k] for k in ('seed','completed','failed_timeout','risk','checks')}),flush=True)
    risks={}
    hours=sum(r['flight_hours'] for r in results)
    for level in ('lowc','nmac'):
        exposure=sum(r['risk'][level]['unordered_pair_seconds'] for r in results)
        risks[level]={'event_count':sum(r['risk'][level]['event_count'] for r in results),
                      'same_corridor_event_count':sum(r['risk'][level]['same_corridor_event_count'] for r in results),
                      'cross_corridor_event_count':sum(r['risk'][level]['cross_corridor_event_count'] for r in results),
                      'events_started_at_admission':sum(r['risk'][level]['events_started_at_admission'] for r in results),
                      'unordered_pair_seconds':exposure, 'directed_pair_seconds':2*exposure,
                      'unordered_seconds_per_flight_hour':exposure/hours,
                      'zero_exposure_scenario_fraction':sum(r['risk'][level]['event_count']==0 for r in results)/len(results),
                      'zero_exposure_flight_fraction':1-sum(r['risk'][level]['aircraft_with_event'] for r in results)/sum(r['planned'] for r in results)}
    summary={'scope':'development-only paper-like no-resolution pilot; no trained policy',
             'config':cfg,'selected_seeds':seeds,'types':types,'scenario_count':len(results),
             'planned':sum(r['planned'] for r in results),'completed':sum(r['completed'] for r in results),
             'failed_timeout':sum(r['failed_timeout'] for r in results),
             'flight_hours':hours,'risk':risks,
             'max_centerline_distance_m':max(r['max_centerline_distance_m'] for r in results),
             'outside_corridor_flights':sum(r['outside_corridor_flights'] for r in results),
             'initialization_wall_seconds':init_seconds,'cases':[str(Path(f'seed-{s}')/'result.json') for s in seeds],
             'all_checks_passed':all(all(r['checks'].values()) for r in results),
             'versions':{p:version(p) for p in ('bluesky-simulator','numpy')}}
    (output/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)
    return 0 if summary['all_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
