"""Constructive finite native-heading demonstrations; not the MARL environment.

The commanded paper lane is recorded separately from the explicit 1 m inward
reference bias. Bends and exits use a return to center; bend TAS is capped at
12 m/s. Route dimensions are constructed from maneuver occupancy. These are
development witnesses, not the old 5 NM cases or a policy-quality comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from lateral_plan import KinematicState, HeadingCommand, native_step, certify_plan

DT = .25
WIDTH = 76.2
ORIGIN = (52., 4.)
RADIUS = 6371000.
GRAVITY = 9.80665
BANK = 25.
DEFECT = 1e-4


def xy(state):
    return ((state.lon_deg-ORIGIN[1])*math.pi/180*RADIUS*math.cos(math.radians(ORIGIN[0])),
            (state.lat_deg-ORIGIN[0])*math.pi/180*RADIUS)


def latlon(point):
    return (ORIGIN[0]+math.degrees(point[1]/RADIUS),
            ORIGIN[1]+math.degrees(point[0]/(RADIUS*math.cos(math.radians(ORIGIN[0])))))


def gap(a, b):
    return (a-b+180.) % 360.-180.


def advance(state, command):
    return native_step(state.lat_deg, state.lon_deg, state.hdg_deg, state.tas_mps,
                       command.hdg_deg, command.tas_mps, dt=DT, accel=3.5,
                       bank_deg=BANK, earth_radius_m=RADIUS, gravity_mps2=GRAVITY)


class ConstructivePlan:
    def __init__(self, initial):
        self.states, self.commands, self.phases, self.events = [initial], [], [], []

    @property
    def current(self):
        return self.states[-1]

    def append(self, heading, speed, phase):
        if len(self.commands) >= 4800:
            raise ValueError('Constructive plan exceeds the unchanged1200s mission horizon')
        command = HeadingCommand(heading % 360., speed)
        state = advance(self.current, command)
        self.commands.append(command)
        self.states.append(state)
        self.phases.append(phase)

    def event(self, event, **fields):
        self.events.append(dict(event=event, tick=len(self.commands), time_s=len(self.commands)*DT, **fields))

    def hold(self, course, speed, seconds, phase, *, decision_align=False):
        for _ in range(math.ceil(seconds/DT)):
            self.append(course, speed, phase)
        if decision_align:
            while len(self.commands) % 20:
                self.append(course, speed, phase)

    def lateral(self, course, speed, line_origin, reference_lane, phase):
        """Finite nonnegative heading lobe; reject any bank-limited proposal.

        Amplitude bisection proposes a route only. The final native forward
        states and full certificate, not the root tolerance, define acceptance.
        """
        radians = math.radians(course)
        right = (math.cos(radians), -math.sin(radians))
        def cross(state):
            p = xy(state)
            return sum((p[j]-line_origin[j])*right[j] for j in (0, 1))
        delta = reference_lane-cross(self.current)
        if abs(delta) < 1e-8:
            return
        sign = math.copysign(1., delta)
        initial = self.current
        def trial(amplitude, n, keep=False):
            state, ratio = initial, 0.
            commands = []
            for j in range(1, n+1):
                angle = amplitude*math.sin(math.pi*j/n)**2 if j < n else 0.
                command = HeadingCommand((course+sign*angle) % 360., speed)
                nxt = advance(state, command)
                cap = math.degrees(DT*GRAVITY*math.tan(math.radians(BANK))/max(nxt.tas_mps, .01))
                ratio = max(ratio, abs(gap(command.hdg_deg, state.hdg_deg))/cap)
                state = nxt
                if keep:
                    commands.append(command)
            return sign*(cross(state)-cross(initial)), ratio, commands
        n = 40
        while n <= 2000:
            span, _, _ = trial(45., n)
            if span >= abs(delta):
                lo, hi = 0., 45.
                for _ in range(42):
                    mid = (lo+hi)/2
                    value, _, _ = trial(mid, n)
                    if value > abs(delta):
                        hi = mid
                    else:
                        lo = mid
                _, ratio, commands = trial((lo+hi)/2, n, True)
                if ratio <= .8:
                    for command in commands:
                        self.append(command.hdg_deg, command.tas_mps, phase)
                    self.event(phase+'_ended', reference_lane_m=reference_lane,
                               achieved_reference_lane_m=cross(self.current), ticks=n,
                               heading_rate_fraction=ratio)
                    return
            n = math.ceil(n*1.25)
        raise ValueError('No bounded monotone lobe proposal passed the native yaw check')


def make_plan(initial, nominal_speed, target_speed, lane_sign, mirror):
    plan = ConstructivePlan(initial)
    lane = lane_sign*(WIDTH-1.)
    plan.hold(90., nominal_speed, 5., 'nominal')
    plan.event('request_target', requested_lane_m=lane_sign*WIDTH, reference_lane_m=lane,
               requested_tas_mps=target_speed)
    plan.lateral(90., target_speed, (0., 0.), lane, 'capture_inbound')
    plan.hold(90., target_speed, 20., 'hold_inbound', decision_align=True)
    plan.event('request_return', requested_lane_m=0.)
    plan.lateral(90., target_speed, (0., 0.), 0., 'return_inbound')
    plan.hold(90., target_speed, 20., 'hold_center_inbound', decision_align=True)
    nominal = [ORIGIN]
    final_course = 90.
    if mirror:
        turn_speed = min(target_speed, 12.)
        slowdown_seconds = abs(plan.current.tas_mps-turn_speed)/3.5+1.
        plan.hold(90., turn_speed, slowdown_seconds, 'bend_decelerate')
        plan.event('bend_started', applied_tas_mps=turn_speed, requested_tas_mps=target_speed)
        increment = math.degrees(DT*GRAVITY*math.tan(math.radians(BANK))/turn_speed)
        n = math.ceil(90./(.75*increment))
        for j in range(1, n+1):
            plan.append(90.+mirror*90.*j/n, turn_speed, 'bend')
        end_turn = xy(plan.current)
        vertex = (end_turn[0], 0.)
        nominal.append(latlon(vertex))
        final_course = (90.+mirror*90.) % 360.
        plan.event('bend_ended', nominal_vertex_xy_m=vertex)
        plan.hold(final_course, target_speed, abs(target_speed-turn_speed)/3.5+1., 'outbound_accelerate', decision_align=True)
        plan.event('request_outbound_target', requested_lane_m=lane_sign*WIDTH, reference_lane_m=lane)
        plan.lateral(final_course, target_speed, vertex, lane, 'capture_outbound')
        plan.hold(final_course, target_speed, 20., 'hold_outbound', decision_align=True)
        plan.event('request_outbound_return', requested_lane_m=0.)
        plan.lateral(final_course, target_speed, vertex, 0., 'return_outbound')
        plan.hold(final_course, target_speed, 20., 'hold_center_outbound', decision_align=True)
    last = xy(plan.current)
    unit = (math.sin(math.radians(final_course)), math.cos(math.radians(final_course)))
    endpoint = tuple(last[j]+max(200., target_speed*25.)*unit[j] for j in (0, 1))
    # Centered endpoint, exactly on the final nominal axis. Do not carry tiny
    # lobe root residuals into the nominal geometry.
    endpoint = (endpoint[0], 0.) if not mirror else (vertex[0], endpoint[1])
    nominal.append(latlon(endpoint))
    while sum((xy(plan.current)[j]-endpoint[j])*unit[j] for j in (0, 1)) < 0.:
        plan.append(final_course, target_speed, 'exit_centered')
    # Choose a constructive finite endpoint inside the final reference tick,
    # then certify the full uncertain pre/final side separation below. The old
    # endpoint could lie arbitrarily near the penultimate sample, causing actual
    # flight to cross a tick early. No native sample is removed by this change.
    a,b=xy(plan.states[-2]),xy(plan.states[-1])
    endpoint=((a[0]+b[0])/2,0.) if not mirror else (vertex[0],(a[1]+b[1])/2)
    nominal[-1]=latlon(endpoint)
    plan.event('reference_exit', full_terminal_tick_preserved=True)
    return plan, nominal, endpoint, unit


def certify_final_exit(states, endpoint_latlon, error_bounds, *, axis, direction):
    """Exact rational side checks in the certificate's frozen local frame.

    Every earlier uncertainty box must be strictly before the finite end plane,
    and the final box strictly after it. Combined with containment in the final
    leg capsule this proves valid crossing in the last interval for the linear
    interpolation, conditional on the same state/error hypotheses.
    """
    if axis not in (0,1) or direction not in (-1,1) or len(states)<2 or len(states)!=len(error_bounds):
        raise ValueError('Invalid terminal-certificate dimensions')
    scale=Fraction(math.pi*RADIUS/180.)
    if axis==0:
        scale*=Fraction(math.cos(math.radians(ORIGIN[0])))
    endpoint=Fraction(endpoint_latlon[1-axis])
    intervals=[]
    for state,bound in zip(states,error_bounds):
        coordinate=state.lon_deg if axis==0 else state.lat_deg
        value=direction*(Fraction(coordinate)-endpoint)*scale
        error=Fraction(bound['ex_m' if axis==0 else 'ey_m'])
        intervals.append((value-error,value+error))
    before=max(interval[1] for interval in intervals[:-1])
    after=intervals[-1][0]
    return dict(conditional_final_interval_crossing=before<0<after,
        maximum_previous_upper_along_m=float(before),final_lower_along_m=float(after),
        arithmetic='Exact rational signed-coordinate comparisons with outward certificate axis errors.',
        full_terminal_tick_retained=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--wall-seconds', type=float, default=180.)
    parser.add_argument('--case-limit', type=int)
    args = parser.parse_args(argv)
    if os.environ.get('LAB_RUN_DIR') != '/output' or Path.cwd() != Path('/workspace'):
        raise ValueError('Use the isolated lab launcher')
    if not 0 < args.wall_seconds <= 1200:
        raise ValueError('Invalid wall budget')
    started = time.perf_counter()
    output = Path('/output')
    from bluesky_diagnostic import initialise
    from paper_performance import install_performance, load_types
    from nr_pilot import centerline_distance_m
    from route_completion import finite_exit_crossing
    types = load_types('configs/uav_types.json')
    bs = initialise(Path(tempfile.mkdtemp(prefix='lateral-native-', dir='/tmp')))
    performance = install_performance(bs, types)
    from bluesky.core import simtime
    from bluesky.traffic.asas import ConflictDetection, ConflictResolution
    from bluesky.tools.aero import tas2cas, g0, Rearth
    if g0 != GRAVITY or Rearth != RADIUS:
        raise ValueError('Native constants differ from the reference')
    kinds = ('Mavic', 'Amzn') if args.suite == 'smoke' else tuple(types)
    declared_count = len(kinds)*12
    selected_count = declared_count if args.case_limit is None else args.case_limit
    if not 1 <= selected_count <= declared_count:
        raise ValueError('case-limit must select a nonempty declared prefix')
    result = dict(schema='bluesky.lateral-reference-probe.v1', suite=args.suite, cases=[],
        declared_count=declared_count, selected_count=selected_count,
        all_checks_passed=False, validation_complete=False, promoted=False,
        scope='Constructive cardinal finite single-aircraft demonstrations, not original development scenarios, the paper controller, arbitrary RL actions, or an effectiveness comparison.',
        execution_changes=dict(lane_inward_reference_bias_m=1., bend_tas_cap_mps=12.,
            center_return_before_bend_and_exit=True, direct_ap_heading_every_s=DT,
            native_lnav=False, finite_plans_precomputed=True, altitude_m=106.68),
        certificate_conditions=dict(position_defect_each_axis_m_per_tick=DEFECT,
            initial_position_error_each_axis_m=1e-6, latitude_band_deg=[51.8,52.2],
            numerical_defect_bound_formally_established=False,
            interval_curve='Linear interpolation of native stored position samples only'),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('src/lateral_plan.py','src/lateral_reference_probe.py','configs/uav_types.json')})
    def save():
        result['wall_seconds'] = time.perf_counter()-started
        temp = output/'result.tmp'
        temp.write_text(json.dumps(result, allow_nan=False, indent=2)+'\n')
        temp.replace(output/'result.json')
    save()
    with gzip.open(output/'traces.jsonl.gz', 'wt', encoding='utf-8') as traces, gzip.open(output/'certificates.jsonl.gz','wt',encoding='utf-8') as certs:
        for kind in kinds:
            for ratio in (.5, 1.05):
                for lane_sign in (-1, 1):
                    for mirror in (0, -1, 1):
                        if len(result['cases']) == selected_count:
                            result.update(validation_complete=True,all_checks_passed=all(c['certificate_passed'] and c.get('all_native_checks_passed',False) for c in result['cases']))
                            save()
                            return 0 if result['all_checks_passed'] else 1
                        if time.perf_counter()-started >= args.wall_seconds:
                            raise TimeoutError('Wall budget before fixture')
                        case_id = f'{kind}-v{ratio}-lane{lane_sign:+d}-turn{mirror:+d}'
                        bs.sim.reset()
                        performance.select()
                        ConflictDetection.setmethod('OFF')
                        ConflictResolution.setmethod('OFF')
                        bs.traf.wind.clear()
                        bs.traf.setnoise(False)
                        simtime.setdt(DT)
                        altitude = 106.68
                        nominal_speed = types[kind]['nominal_tas_mps']
                        target_speed = nominal_speed*ratio
                        if bs.traf.cre('F001', kind, *ORIGIN, 90., altitude, float(tas2cas(nominal_speed, altitude))) is not True:
                            raise RuntimeError('Native creation failed')
                        bs.traf.ap.selaltcmd(0, altitude, 0.)
                        bs.traf.ap.selhdgcmd(0, 90.)
                        bs.traf.swvnav[0] = bs.traf.swvnavspd[0] = False
                        if bs.traf.ap.turnphi[0] != 0. or abs(math.degrees(bs.traf.ap.bankdef[0])-BANK)>1e-10:
                            raise RuntimeError('Unexpected native bank mode')
                        def current():
                            return KinematicState(float(bs.traf.lat[0]),float(bs.traf.lon[0]),float(bs.traf.hdg[0]),float(bs.traf.tas[0]))
                        initial = current()
                        plan, nominal, endpoint, final_unit = make_plan(initial, nominal_speed, target_speed, lane_sign, mirror)
                        certificate = certify_plan(plan.states, plan.commands, nominal, origin_latlon=ORIGIN,
                            half_width_m=WIDTH, latitude_band_deg=(51.8,52.2), initial_error_xy_m=(1e-6,1e-6),
                            per_step_defect_m=DEFECT, dt=DT, accel=3.5, bank_deg=BANK,
                            earth_radius_m=RADIUS, gravity_mps2=GRAVITY,
                            speed_uncertainty_mps=2e-4, scoring_roundoff_m=1e-6)
                        certs.write(json.dumps(dict(case_id=case_id,certificate=certificate),allow_nan=False)+'\n')
                        terminal_certificate=certify_final_exit(plan.states,nominal[-1],certificate['error_bounds'],
                            axis=1 if mirror else 0,direction=-mirror if mirror else 1)
                        terminal_certificate['final_interval_in_final_leg_capsule']=(
                            certificate['intervals'][-1]['nominal_segment_index']==len(nominal)-2)
                        row = dict(case_id=case_id,type=kind,speed_ratio=ratio,lane_sign=lane_sign,mirror=mirror,
                            nominal_waypoints_latlon=nominal, reference_steps=len(plan.commands), reference_duration_s=len(plan.commands)*DT,
                            events=plan.events, certificate_passed=certificate['conditional_certificate'] and terminal_certificate['conditional_final_interval_crossing'] and terminal_certificate['final_interval_in_final_leg_capsule'],
                            terminal_certificate=terminal_certificate,
                            certificate_failures=certificate['failures'], native_executed=False)
                        result['cases'].append(row)
                        if not row['certificate_passed']:
                            save()
                            print(json.dumps(dict(case_id=case_id,certificate=False,failures=row['certificate_failures'][:3])),flush=True)
                            continue
                        traces.write(json.dumps(dict(case_id=case_id,tick=0,actual=asdict(initial),reference=asdict(plan.states[0])))+'\n')
                        bs.sim.op()
                        max_distance=max_error=max_local_defect=max_tas_error=max_heading_error=max_accel=max_yaw_ratio=0.
                        outside=0
                        actual_exit=None
                        hold_samples=hold_failed=center_hold_samples=center_hold_failed=tube_failed=0
                        prior=initial
                        for k,(command,reference,phase) in enumerate(zip(plan.commands,plan.states[1:],plan.phases),1):
                            if time.perf_counter()-started >= args.wall_seconds:
                                raise TimeoutError('Wall budget within fixture')
                            bs.traf.ap.selspdcmd(0,float(tas2cas(command.tas_mps,float(bs.traf.alt[0]))))
                            bs.traf.ap.selhdgcmd(0,command.hdg_deg)
                            bs.sim.step()
                            actual=current()
                            # Local one-step consistency is observed at the actual
                            # prior state, independently of accumulated deviation.
                            predicted=advance(KinematicState(prior.lat_deg,prior.lon_deg,
                                reference.hdg_deg,reference.tas_mps),
                                HeadingCommand(reference.hdg_deg,reference.tas_mps))
                            ax,ay=xy(actual)
                            rx,ry=xy(reference)
                            px,py=xy(predicted)
                            err=math.hypot(ax-rx,ay-ry)
                            local=max(abs(ax-px),abs(ay-py))
                            tube=certificate['error_bounds'][k]
                            tube_failed+=abs(ax-rx)>tube['ex_m'] or abs(ay-ry)>tube['ey_m']
                            distance=centerline_distance_m((actual.lat_deg,actual.lon_deg),nominal)
                            actual_yaw=abs(gap(actual.hdg_deg,prior.hdg_deg))
                            cap=math.degrees(DT*GRAVITY*math.tan(math.radians(BANK))/max(actual.tas_mps,.01))
                            max_distance=max(max_distance,distance)
                            max_error=max(max_error,err)
                            max_local_defect=max(max_local_defect,local)
                            max_tas_error=max(max_tas_error,abs(actual.tas_mps-reference.tas_mps))
                            max_heading_error=max(max_heading_error,abs(gap(actual.hdg_deg,command.hdg_deg)))
                            max_accel=max(max_accel,abs(actual.tas_mps-prior.tas_mps)/DT)
                            max_yaw_ratio=max(max_yaw_ratio,actual_yaw/cap)
                            outside+=distance>WIDTH
                            if phase in ('hold_inbound','hold_outbound','hold_center_inbound','hold_center_outbound'):
                                centered='center' in phase
                                inbound=phase.endswith('inbound')
                                course=90. if inbound else (90.+mirror*90.)%360.
                                base=(0.,0.) if inbound else xy(KinematicState(*nominal[1],0.,0.))
                                right=(math.cos(math.radians(course)),-math.sin(math.radians(course)))
                                cross=sum(((ax,ay)[j]-base[j])*right[j] for j in (0,1))
                                failed=abs(cross-(0. if centered else lane_sign*WIDTH))>2. or abs(gap(actual.hdg_deg,course))>5. or abs(actual.tas_mps-target_speed)>1e-3
                                if centered:
                                    center_hold_samples+=1
                                    center_hold_failed+=failed
                                else:
                                    hold_samples+=1
                                    hold_failed+=failed
                            detail=finite_exit_crossing((*xy(prior),altitude),(ax,ay,float(bs.traf.alt[0])),endpoint,final_unit,WIDTH,76.2,137.16)
                            if detail is not None and actual_exit is None:
                                actual_exit=dict(tick=k,**detail)
                            traces.write(json.dumps(dict(case_id=case_id,tick=k,time_s=float(bs.sim.simt),phase=phase,
                                command=asdict(command),reference=asdict(reference),actual=asdict(actual),
                                distance_m=distance,local_position_defect_m=local,altitude_m=float(bs.traf.alt[0]),vs_mps=float(bs.traf.vs[0])),allow_nan=False)+'\n')
                            if abs(float(bs.sim.simt)-k*DT)>1e-8 or float(bs.traf.alt[0])!=altitude or float(bs.traf.vs[0])!=0.:
                                raise RuntimeError('Unexpected time or fixed-altitude update')
                            if bs.traf.wind.winddim or bs.traf.swlnav[0] or bs.traf.ap.turnphi[0]!=0.:
                                raise RuntimeError('Native execution mode changed')
                            prior=actual
                        row.update(native_executed=True, actual_exit=actual_exit,
                            max_centerline_distance_m=max_distance, raw_outside_seconds=outside*DT,
                            max_reference_error_m=max_error,max_local_position_defect_m=max_local_defect,
                            max_tas_reference_error_mps=max_tas_error,max_command_heading_error_deg=max_heading_error,
                            max_tas_acceleration_mps2=max_accel,max_actual_yaw_cap_ratio=max_yaw_ratio,
                            target_hold_samples=hold_samples,target_hold_failed_samples=hold_failed,
                            center_hold_samples=center_hold_samples,center_hold_failed_samples=center_hold_failed,
                            samples_outside_declared_position_tube=tube_failed,
                            native_checks=dict(raw_sampled_containment=outside==0,
                                observed_local_defect_within_hypothesis=max_local_defect<=DEFECT,
                                observed_speed_error_within_hypothesis=max_tas_error<=2e-4,
                                observed_positions_in_declared_tube=tube_failed==0,
                                heading_commands_realized=max_heading_error<=1e-9,
                                tas_acceleration_within_table=max_accel<=3.5+1e-8,
                                yaw_within_native_bank=max_yaw_ratio<=1+1e-8,
                                complete_target_holds=hold_failed==0 and hold_samples>0,
                                complete_center_holds=center_hold_failed==0 and center_hold_samples>0,
                                genuine_final_exit=bool(actual_exit and actual_exit['within_width'] and actual_exit['within_height']),
                                terminal_sample_is_first_exit=bool(actual_exit and actual_exit['tick']==len(plan.commands))))
                        row['all_native_checks_passed']=all(row['native_checks'].values())
                        save()
                        print(json.dumps(dict(case_id=case_id,certificate=True,native_ok=row['all_native_checks_passed'],max_distance_m=max_distance,max_reference_error_m=max_error)),flush=True)
        result.update(validation_complete=True,all_checks_passed=all(c['certificate_passed'] and c.get('all_native_checks_passed',False) for c in result['cases']))
        save()
    return 0 if result['all_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
