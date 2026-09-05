"""Experimental finite heading plans on an immutable original nominal polyline.

This pure module neither imports BlueSky nor changes native physical state. It
constructs a local tangent-circle/quintic path, solves each native inverse chord,
then records only ``native_step`` results. Successful planning is NOT a containment
certificate: callers must independently certify every retained interval and the
native realization hypotheses. Positive lane is right of the original first leg.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from lateral_plan import (GRAVITY_MPS2, PROJECT_RADIUS_M, HeadingCommand,
                          KinematicState, inverse_chord, local_xy, native_step)

RADIUS_M = 70.
CORNER_TAS_MPS = 12.
ACCEL_MPS2 = 3.5
BANK_DEG = 25.
LANE_INSET_M = 1.
HALF_WIDTH_M = 76.2
HOLD_SECONDS = 20.
SPEED_UNCERTAINTY_MPS = .0002
MAX_PATH_RESIDUAL_M = 1e-4
MAX_STEPS = 20000


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1])


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1])


def _mul(a, x):
    return (a[0]*x, a[1]*x)


def _dot(a, b):
    return a[0]*b[0]+a[1]*b[1]


def _norm(a):
    return math.hypot(*a)


def _angle(a, b):
    return (a-b+180.) % 360.-180.


@dataclass(frozen=True)
class _Piece:
    start: float
    end: float
    kind: str
    a: tuple
    b: tuple
    leg: int
    center: tuple | None = None
    radius: float = 0.
    angle0: float = 0.
    sweep: float = 0.

    def point(self, station):
        fraction = max(0., min(1., (station-self.start)/(self.end-self.start)))
        if fraction == 0.:
            return self.a
        if fraction == 1.:
            return self.b
        if self.kind == 'line':
            return _add(self.a, _mul(_sub(self.b, self.a), fraction))
        angle = self.angle0+self.sweep*fraction
        return _add(self.center, (self.radius*math.cos(angle), self.radius*math.sin(angle)))


class _Path:
    """Station is filleted-centerline arclength, longitudinal during lane shifts."""
    def __init__(self, nominal):
        self.nominal = tuple(tuple(float(x) for x in p) for p in nominal)
        if not 2 <= len(self.nominal) <= 100:
            raise ValueError('Requires between2 and100 fixed vertices')
        self.origin = self.nominal[0]
        self.xy = tuple(local_xy(p, self.origin) for p in self.nominal)
        if any(abs(b[1]-a[1]) >= 180 for a, b in zip(self.nominal, self.nominal[1:])):
            raise ValueError('Unwrapped local longitude is required')
        self.lengths = tuple(_norm(_sub(b, a)) for a, b in zip(self.xy, self.xy[1:]))
        if min(self.lengths) <= 1e-6:
            raise ValueError('Zero-length original segment')
        self.units = tuple(_mul(_sub(b, a), 1./length)
                           for a, b, length in zip(self.xy, self.xy[1:], self.lengths))
        self.right = (self.units[0][1], -self.units[0][0])
        self.scale = math.pi*PROJECT_RADIUS_M/180.
        self.cos_origin = math.cos(math.radians(self.origin[0]))
        # This independent route-derived band is only a planning metric bound;
        # the external certificate supplies/checks its own latitude band.
        self.band = (min(p[0] for p in self.nominal)-.005,
                     max(p[0] for p in self.nominal)+.005)
        if not -89 < self.band[0] <= self.band[1] < 89:
            raise ValueError('Planning latitude band exceeds the nonpolar domain')
        self.stretch = max(1., self.cos_origin/min(math.cos(math.radians(x)) for x in self.band))
        fillets = []
        for index in range(1, len(self.xy)-1):
            incoming, outgoing = self.units[index-1:index+1]
            alpha = math.atan2(incoming[0]*outgoing[1]-incoming[1]*outgoing[0],
                               _dot(incoming, outgoing))
            if abs(alpha) > math.pi/2+1e-9:
                raise ValueError('Original turn exceeds the declared90-degree domain')
            if abs(alpha) < 1e-10:
                fillets.append(None)
                continue
            radius = min(RADIUS_M, .2*min(self.lengths[index-1:index+1])/math.tan(abs(alpha)/2))
            distance = radius*math.tan(abs(alpha)/2)
            vertex = self.xy[index]
            a, b = _sub(vertex, _mul(incoming, distance)), _add(vertex, _mul(outgoing, distance))
            left = (-incoming[1], incoming[0])
            center = _add(a, _mul(left, math.copysign(radius, alpha)))
            fillets.append(dict(vertex_index=index, radius_m=radius, tangent_distance_m=distance,
                signed_angle_deg=math.degrees(alpha), incoming_xy=a, outgoing_xy=b,
                center_xy=center, angle0=math.atan2(a[1]-center[1], a[0]-center[0]), sweep=alpha))
        pieces, station, previous = [], 0., self.xy[0]
        for leg in range(len(self.lengths)):
            corner = fillets[leg] if leg < len(fillets) else None
            end = corner['incoming_xy'] if corner else self.xy[leg+1]
            length = _norm(_sub(end, previous))
            if length > 1e-8:
                pieces.append(_Piece(station, station+length, 'line', previous, end, leg))
                station += length
            if corner:
                length = corner['radius_m']*abs(corner['sweep'])
                pieces.append(_Piece(station, station+length, 'arc', end, corner['outgoing_xy'], leg,
                    corner['center_xy'], corner['radius_m'], corner['angle0'], corner['sweep']))
                corner['path_start_m'], corner['path_end_m'] = station, station+length
                station += length
                previous = corner['outgoing_xy']
            else:
                previous = end
        self.pieces = tuple(pieces)
        self.fillets = tuple(c for c in fillets if c)
        self.length = station
        self.first_leg_end = self.pieces[0].end
        self.final_outbound = max((p.end for p in pieces if p.kind == 'arc'), default=0.)
        self.lane = None

    def latlon(self, point):
        return (self.origin[0]+point[1]/self.scale,
                self.origin[1]+point[0]/(self.scale*self.cos_origin))

    def piece(self, station):
        return next((p for p in self.pieces if station < p.end), self.pieces[-1])

    def lane_at(self, station):
        if self.lane is None:
            return 0.
        a, b, c, d, value = self.lane
        if station <= a or station >= d:
            return 0.
        if b <= station <= c:
            return value
        fraction = (station-a)/(b-a) if station < b else (d-station)/(d-c)
        return value*fraction**3*(10.+fraction*(-15.+6.*fraction))

    def point(self, station):
        if station > self.length:
            return _add(self.xy[-1], _mul(self.units[-1], station-self.length))
        result = self.piece(station).point(station)
        lane = self.lane_at(station)
        return _add(result, _mul(self.right, lane)) if lane else result

    def phase(self, old_station, new_station):
        if self.lane:
            a, b, c, d, _ = self.lane
            # Hold comprises complete sampled intervals on the constant lane.
            if old_station >= b and new_station <= c:
                return 'lane_hold'
            if a < new_station and old_station < b:
                return 'lane_capture'
            if c < new_station and old_station < d:
                return 'lane_return'
        if any(old_station < p.end and new_station > p.start for p in self.pieces if p.kind == 'arc'):
            return 'turn'
        return 'center'

    def progress(self, state, station, residual, speed_cap):
        piece = self.piece(station)
        leg = piece.leg
        if piece.kind == 'arc' and station >= (piece.start+piece.end)/2:
            leg += 1
        xy = local_xy(state, self.origin)
        along = _dot(_sub(xy, self.xy[leg]), self.units[leg])
        return dict(nominal_leg_index=leg, path_station_m=station, path_error_m=residual,
                    reference_lane_m=self.lane_at(station), nominal_along_m=along,
                    nominal_leg_length_m=self.lengths[leg], speed_cap_active=speed_cap)


def plan_fixed_route(initial, nominal_latlon, nominal_tas_mps, *, lane_m=0.,
                     request_age_s=5., dt=.25, timeout_s=1200.):
    """Build an explicitly experimental finite plan; scientific failures are data.

    ``states``/``progress`` include the initial state. ``commands``/``phases`` have
    one entry per transition. Leg indices are zero-based. ``lane_hold`` denotes
    complete observed intervals of the constant reference lane, not a continuous
    physical proof. Rejected first-leg requests continue on the nominal path.
    The fixed endpoint is unchanged; two additional reference ticks are retained
    after the first eligible final-plane crossing for actual native exit timing.
    """
    states, commands, phases, progress = [initial], [], [], []
    lane_request = dict(requested_lane_m=lane_m, reference_lane_m=0., status='not_requested',
                        reason=None, request_age_s=request_age_s, inset_m=LANE_INSET_M,
                        scope='first_original_leg_only', observed_hold_seconds=0.)
    result = dict(success=False, failure=None, states=(), commands=(), phases=(), progress=(),
                  nominal_latlon=(), lane_request=lane_request, geometry={},
                  max_heading_rate_ratio=0., reference_exit_tick=None,
                  continuation_ticks=2, containment_established=False)

    def finish(kind=None, detail=None, tick=None):
        result.update(states=tuple(states), commands=tuple(commands), phases=tuple(phases),
                      progress=tuple(progress), success=kind is None,
                      failure=None if kind is None else dict(kind=kind, detail=detail, tick=tick))
        return result

    try:
        if not isinstance(initial, KinematicState):
            raise ValueError('initial must be KinematicState')
        for value in (nominal_tas_mps, lane_m, request_age_s, dt, timeout_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Finite scalar parameters required')
        if not 0 < nominal_tas_mps <= 85 or not 0 < dt <= .25 or not 0 < timeout_s <= 3600 or request_age_s < 0:
            raise ValueError('Requires TAS in(0,85], dt in(0,.25], timeout in(0,3600], age>=0')
        if abs(request_age_s/dt-round(request_age_s/dt)) > 1e-8:
            raise ValueError('request_age_s must align with the native dt grid')
        path = _Path(nominal_latlon)
        result['nominal_latlon'] = path.nominal
        if _norm(local_xy(initial, path.origin)) > MAX_PATH_RESIDUAL_M:
            return finish('initial_position_not_route_origin', 'No initial-state repositioning is permitted', 0)
        limit = int(math.floor(timeout_s/dt+1e-9))
        if limit > MAX_STEPS:
            raise ValueError('Requested plan exceeds the20000-step hard cap')
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        return finish('invalid_input_or_geometry', str(exc), 0)

    result['geometry'] = dict(original_vertices_latlon=path.nominal, original_vertices_xy=path.xy,
        original_endpoint_latlon=path.nominal[-1], original_segment_lengths_m=path.lengths,
        fillets=path.fillets, path_length_m=path.length, final_outbound_station_m=path.final_outbound,
        first_tangent_station_m=path.first_leg_end, planning_latitude_band_deg=path.band,
        station_definition='filleted centerline arclength; longitudinal parameter during first-leg quintic',
        corner_tas_cap_mps=min(CORNER_TAS_MPS, nominal_tas_mps), bank_deg=BANK_DEG,
        acceleration_mps2=ACCEL_MPS2, heading_speed_margin_mps=SPEED_UNCERTAINTY_MPS,
        endpoint_modified=False, original_vertices_modified=False)
    station = 0.
    progress.append(path.progress(initial, station, 0., False))
    request_tick = round(request_age_s/dt)
    if lane_m:
        lane_request['status'] = 'pending'
    cap = min(CORNER_TAS_MPS, nominal_tas_mps)
    turn_factor = math.degrees(GRAVITY_MPS2*math.tan(math.radians(BANK_DEG)))

    for tick in range(limit):
        old = states[-1]
        if lane_m and tick == request_tick:
            reference_lane = math.copysign(max(0., abs(lane_m)-LANE_INSET_M), lane_m)
            # max sigma'' =10sqrt(3)/3; reserve half the native turning acceleration.
            ramp = max(50., nominal_tas_mps*4., math.sqrt((10*math.sqrt(3)/3)*abs(reference_lane)
                        *nominal_tas_mps**2/(.5*GRAVITY_MPS2*math.tan(math.radians(BANK_DEG)))))
            hold = nominal_tas_mps*(HOLD_SECONDS+2*dt)*path.stretch
            end = station+2*ramp+hold
            braking = max(0., (nominal_tas_mps**2-cap**2)/(2*ACCEL_MPS2)) if path.fillets else 0.
            reserve = path.stretch*(braking+2*nominal_tas_mps*dt)+10.
            lane_request.update(reference_lane_m=reference_lane, request_station_m=station,
                ramp_longitudinal_m=ramp, hold_longitudinal_m=hold,
                return_end_station_m=end, braking_and_exit_reserve_m=reserve,
                available_first_leg_m=path.first_leg_end-station)
            if abs(lane_m) > HALF_WIDTH_M:
                lane_request.update(status='rejected', reason='requested_lane_exceeds_original_half_width')
            elif end+reserve >= path.first_leg_end:
                lane_request.update(status='rejected', reason='complete_shift_hold_return_and_braking_do_not_fit_first_leg')
            else:
                lane_request.update(status='accepted', reason=None)
                path.lane = (station, station+ramp, station+ramp+hold, end, reference_lane)

        target_speed = nominal_tas_mps
        for piece in path.pieces:
            if piece.kind != 'arc' or piece.end <= station:
                continue
            future_speed = min(nominal_tas_mps, old.tas_mps+2*ACCEL_MPS2*dt)
            brake = max(0., (future_speed**2-cap**2)/(2*ACCEL_MPS2))
            reserve = path.stretch*(brake+2*future_speed*dt)+10.
            if piece.start-station <= reserve:
                target_speed = cap
            break
        if result['reference_exit_tick'] is not None:
            target_speed = nominal_tas_mps
        speed_change = max(-ACCEL_MPS2*dt, min(ACCEL_MPS2*dt, target_speed-old.tas_mps))
        next_speed = old.tas_mps+speed_change
        travel = dt*next_speed

        def chord_at(value):
            return inverse_chord(old, path.latlon(path.point(value)), dt=dt)

        try:
            lo, hi = station, station+max(1., 2*travel+2.)
            if chord_at(lo).tas_mps*dt >= travel:
                return finish('path_residual_prevents_local_forward_root', dict(station_m=station), tick)
            for _ in range(12):
                if chord_at(hi).tas_mps*dt >= travel:
                    break
                hi = station+2*(hi-station)
            else:
                return finish('native_inverse_chord_not_bracketed', dict(station_m=station), tick)
            for _ in range(42):
                middle = (lo+hi)/2
                if chord_at(middle).tas_mps*dt < travel:
                    lo = middle
                else:
                    hi = middle
            next_station = (lo+hi)/2
            chord = chord_at(next_station)
            demand = abs(_angle(chord.hdg_deg, old.hdg_deg))
            available = dt*turn_factor/max(next_speed+SPEED_UNCERTAINTY_MPS, .01)
            ratio = demand/available
            result['max_heading_rate_ratio'] = max(result['max_heading_rate_ratio'], ratio)
            if ratio > .98:
                return finish('heading_command_not_robustly_reachable',
                    dict(demand_deg=demand, available_deg=available, ratio=ratio,
                         station_m=station, target_speed_mps=target_speed), tick)
            command = HeadingCommand(chord.hdg_deg, target_speed)
            new = native_step(old.lat_deg, old.lon_deg, old.hdg_deg, old.tas_mps,
                              command.hdg_deg, command.tas_mps, dt=dt)
            residual = _norm(_sub(local_xy(new, path.origin), path.point(next_station)))
            phase = path.phase(station, next_station)
            commands.append(command)
            states.append(new)
            phases.append(phase)
            progress.append(path.progress(new, next_station, residual, target_speed < nominal_tas_mps))
            if residual > MAX_PATH_RESIDUAL_M:
                return finish('native_reference_path_residual_exceeded',
                              dict(residual_m=residual, threshold_m=MAX_PATH_RESIDUAL_M), tick+1)
            if not path.band[0] <= new.lat_deg <= path.band[1]:
                return finish('reference_left_predeclared_planning_band', dict(latitude_deg=new.lat_deg), tick+1)
            if phase == 'lane_hold':
                lane_request['observed_hold_seconds'] += dt
            old_station, station = station, next_station
            old_along = _dot(_sub(local_xy(old, path.origin), path.xy[-1]), path.units[-1])
            new_along = _dot(_sub(local_xy(new, path.origin), path.xy[-1]), path.units[-1])
            if (result['reference_exit_tick'] is None and station >= path.final_outbound
                    and old_station >= path.final_outbound and old_along < 0 <= new_along):
                result['reference_exit_tick'] = tick+1
                result['reference_exit_along_m'] = new_along
            if result['reference_exit_tick'] is not None and tick+1 >= result['reference_exit_tick']+2:
                if lane_request['status'] == 'pending':
                    lane_request.update(status='rejected', reason='request_age_after_reference_exit')
                if lane_request['status'] == 'accepted' and lane_request['observed_hold_seconds'] < HOLD_SECONDS:
                    return finish('accepted_lane_hold_shorter_than_declared_duration',
                                  dict(observed_seconds=lane_request['observed_hold_seconds']), tick+1)
                return finish()
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            return finish('native_inverse_or_geometry_numerical_failure', str(exc), tick)
    return finish('timeout_before_fixed_exit_and_complete_continuation',
                  dict(timeout_s=timeout_s, last_station_m=station, path_length_m=path.length), len(commands))
