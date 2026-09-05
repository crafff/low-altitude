"""Pure experimental lateral prediction and a conditional finite interpolation certificate.

No BlueSky imports. Constants below match installed BlueSky1.1.1 tools/aero.py.
The scalar predictor matches the native real-arithmetic map, not necessarily the
last bit of NumPy/libm evaluation. Supplied defect bounds are hypotheses, never
estimated from replay residuals. The certificate covers straight interpolation
between physical ticks; it does not supply a native continuous-time trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math


EARTH_RADIUS_M = 6371000.
PROJECT_RADIUS_M = 6371000.
GRAVITY_MPS2 = 9.80665
MINIMUM_TURN_TAS_MPS = .01
MAXIMUM_STEPS = 100000
# Explicit rational enclosure of pi, used only for certificate trigonometric bounds.
PI_LO = Fraction('3.14159265358979323846264338327950288419716939937510')
PI_HI = Fraction('3.14159265358979323846264338327950288419716939937511')


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    value = float(value)
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f'Invalid sign for {name}')
    return value


def _latlon(point):
    if isinstance(point, KinematicState):
        return point.lat_deg, point.lon_deg
    if len(point) != 2:
        raise ValueError('Expected a latitude/longitude pair')
    lat, lon = (_number(value, 'coordinate') for value in point)
    if not -89 < lat < 89 or not -180 <= lon <= 180:
        raise ValueError('Requires the nonpolar, unwrapped project coordinate domain')
    return lat, lon


@dataclass(frozen=True)
class KinematicState:
    lat_deg: float
    lon_deg: float
    hdg_deg: float
    tas_mps: float

    def __post_init__(self):
        _latlon((self.lat_deg, self.lon_deg))
        _number(self.hdg_deg, 'heading')
        _number(self.tas_mps, 'TAS', nonnegative=True)


@dataclass(frozen=True)
class HeadingCommand:
    hdg_deg: float
    tas_mps: float

    def __post_init__(self):
        _number(self.hdg_deg, 'command heading')
        _number(self.tas_mps, 'resolved command TAS', nonnegative=True)


@dataclass(frozen=True)
class ChordVelocity:
    north_mps: float
    east_mps: float
    hdg_deg: float
    tas_mps: float


def _parameters(dt, accel, bank_deg, earth_radius_m, gravity_mps2, minimum_turn_tas_mps):
    values = tuple(_number(value, name, positive=True) for value, name in (
        (dt, 'dt'), (accel, 'accel'), (bank_deg, 'bank_deg'), (earth_radius_m, 'earth_radius_m'),
        (gravity_mps2, 'gravity_mps2'), (minimum_turn_tas_mps, 'minimum_turn_tas_mps')))
    if values[2] >= 90:
        raise ValueError('Bank angle must be strictly below90 degrees')
    return values


def native_step(lat_deg, lon_deg, hdg_deg, tas_mps, command_hdg_deg, command_tas_mps,
                *, dt=.25, accel=3.5, bank_deg=25., earth_radius_m=EARTH_RADIUS_M,
                gravity_mps2=GRAVITY_MPS2, minimum_turn_tas_mps=MINIMUM_TURN_TAS_MPS):
    """Scalar native no-wind TAS→heading→latitude→longitude update.

    command_tas_mps is the already resolved/clipped TAS intent, not a CAS API
    request. No wind, ASAS, turnphi override, vertical model or route logic is
    included. Zero TAS uses the native0.01m/s denominator for turning.
    """
    state = KinematicState(lat_deg, lon_deg, hdg_deg, tas_mps)
    command = HeadingCommand(command_hdg_deg, command_tas_mps)
    dt, accel, bank_deg, earth_radius_m, gravity_mps2, minimum_turn_tas_mps = _parameters(
        dt, accel, bank_deg, earth_radius_m, gravity_mps2, minimum_turn_tas_mps)
    delta = command.tas_mps-state.tas_mps
    speed = state.tas_mps+math.copysign(accel*dt, delta) if abs(delta) > abs(dt*accel) else command.tas_mps
    turnrate = math.degrees(gravity_mps2*math.tan(math.radians(bank_deg))/max(speed, minimum_turn_tas_mps))
    difference = (command.hdg_deg-state.hdg_deg+180.) % 360.-180.
    heading = (state.hdg_deg+math.copysign(dt*turnrate, difference)
               if abs(difference) > abs(dt*turnrate) else command.hdg_deg) % 360.
    north, east = speed*math.cos(math.radians(heading)), speed*math.sin(math.radians(heading))
    latitude = state.lat_deg+math.degrees(dt*north/earth_radius_m)
    longitude = state.lon_deg+math.degrees(dt*east/math.cos(math.radians(latitude))/earth_radius_m)
    return KinematicState(latitude, longitude, heading, speed)


def inverse_chord(start, end, *, dt=.25, earth_radius_m=EARTH_RADIUS_M):
    """Invert the native position map using cos(end latitude), not mean/start latitude.

    This supplies a desired chord velocity; it does not certify acceleration or
    bank reachability. A zero chord reports heading0 by convention.
    """
    lat0, lon0 = _latlon(start)
    lat1, lon1 = _latlon(end)
    dt = _number(dt, 'dt', positive=True)
    radius = _number(earth_radius_m, 'earth_radius_m', positive=True)
    if abs(lon1-lon0) >= 180:
        raise ValueError('Longitude wrapping is outside this local inverse')
    north = radius*math.radians(lat1-lat0)/dt
    east = radius*math.cos(math.radians(lat1))*math.radians(lon1-lon0)/dt
    return ChordVelocity(north, east, math.degrees(math.atan2(east, north)) % 360., math.hypot(north, east))


def simulate_plan(initial_state, commands, **parameters):
    """Return immutable initial+poststep states for a fixed immutable command sequence."""
    if not isinstance(initial_state, KinematicState):
        raise ValueError('Expected KinematicState initial_state')
    commands = tuple(commands)
    if not 1 <= len(commands) <= MAXIMUM_STEPS or any(not isinstance(item, HeadingCommand) for item in commands):
        raise ValueError('Expected a bounded, nonempty HeadingCommand sequence')
    states = [initial_state]
    for command in commands:
        old = states[-1]
        states.append(native_step(old.lat_deg, old.lon_deg, old.hdg_deg, old.tas_mps,
                                  command.hdg_deg, command.tas_mps, **parameters))
    return tuple(states)


def local_xy(point, origin_latlon, *, local_radius_m=PROJECT_RADIUS_M):
    point, origin = _latlon(point), _latlon(origin_latlon)
    scale = math.pi*_number(local_radius_m, 'local_radius_m', positive=True)/180.
    return ((point[1]-origin[1])*scale*math.cos(math.radians(origin[0])), (point[0]-origin[0])*scale)


def raw_centerline_distance_m(point, nominal_latlon):
    """Retain nr_pilot.centerline_distance_m's current-latitude finite-polyline metric."""
    latitude, longitude = _latlon(point)
    vertices = tuple(_latlon(value) for value in nominal_latlon)
    if len(vertices) < 2:
        raise ValueError('A nominal route requires at least two vertices')
    scale = math.pi*PROJECT_RADIUS_M/180.
    positions = [((lon-longitude)*scale*math.cos(math.radians(latitude)),
                  (lat-latitude)*scale) for lat, lon in vertices]
    distances = []
    for a, b in zip(positions, positions[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        if dx*dx+dy*dy == 0:
            raise ValueError('Zero-length nominal segment')
        fraction = max(0, min(1, -(a[0]*dx+a[1]*dy)/(dx*dx+dy*dy)))
        distances.append(math.hypot(a[0]+fraction*dx, a[1]+fraction*dy))
    return min(distances)


@lru_cache(maxsize=8192)
def _cos_bounds(degrees):
    """Rational alternating-series bounds, including the explicit pi enclosure."""
    degrees = abs(Fraction(degrees))
    if degrees > 90:
        raise ValueError('Certificate trigonometry is restricted to0..90 degrees')
    low_angle, high_angle = degrees*PI_LO/180, degrees*PI_HI/180
    def partial(angle, last):
        return sum(((-1)**k*angle**(2*k)/math.factorial(2*k) for k in range(last+1)), Fraction(0))
    # The omitted tail decreases on this domain; even truncation is above
    # cosine and odd truncation below (the first two terms need not decrease).
    lower = partial(high_angle, 9)
    upper = partial(low_angle, 8)
    return max(Fraction(0), lower), min(Fraction(1), upper)


@lru_cache(maxsize=8192)
def _sin_upper(degrees):
    angle = abs((Fraction(degrees)+180) % 360-180)
    acute = min(angle, 180-angle)
    return _cos_bounds(Fraction(90)-Fraction(acute))[1]


def _ceil_float(value):
    """Outward conversion of an exact rational bound to a printable float."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Certificate arithmetic exceeded finite float output')
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _sqrt_upper(value):
    # An exact rational enclosure on a1e-12m grid, not a floating sqrt comparison.
    scale = 10**12
    root = math.isqrt(value.numerator*scale*scale//value.denominator)
    if Fraction(root*root, scale*scale) < value:
        root += 1
    return Fraction(root, scale)


def _grid_upper(value):
    """Exact outward1e-12m rounding bounds denominator growth during recursion."""
    scale = 10**12
    integer = (value.numerator*scale+value.denominator-1)//value.denominator
    return Fraction(integer, scale)


def _distance_squared(point, start, end):
    delta = tuple(end[j]-start[j] for j in (0, 1))
    length2 = sum(value*value for value in delta)
    if length2 <= 0:
        raise ValueError('Nominal segments must have positive length')
    offset = tuple(point[j]-start[j] for j in (0, 1))
    projection = sum(offset[j]*delta[j] for j in (0, 1))
    if projection <= 0:
        return sum(value*value for value in offset)
    if projection >= length2:
        return sum((point[j]-end[j])**2 for j in (0, 1))
    return sum(value*value for value in offset)-projection*projection/length2


def certify_plan(reference_states, commands, nominal_latlon, *, origin_latlon,
                 half_width_m, latitude_band_deg, initial_error_xy_m, per_step_defect_m,
                 dt=.25, accel=3.5, bank_deg=25., earth_radius_m=EARTH_RADIUS_M,
                 gravity_mps2=GRAVITY_MPS2, minimum_turn_tas_mps=MINIMUM_TURN_TAS_MPS,
                 speed_uncertainty_mps=0., scoring_roundoff_m=0.,
                 initial_error_justification='Externally supplied hypothesis; not established by this module.',
                 defect_justification='Externally supplied hypothesis; not inferred from replay maxima.'):
    """Conditional finite certificate using exact-rational capsule inequalities.

    Error recurrence per axis, in the fixed project frame:
      ey[k+1] = ey[k]+eta[k]
      ex[k+1] = ex[k] + dt*V*|sin(hdg)|*cos(origin)/Rnative
                           * sup|sec(lat)tan(lat)| * ey[k+1] + eta[k].
    Trigonometric factors use rational upper enclosures over a predeclared band.
    The supplied per-axis defect bounds the SUM of the actual-step residual
    against the prescribed-reference-V/heading real map and the stored-reference
    evaluation residual, uniformly over admissible states. This also encloses
    the ideal intermediate next latitudes used in the mean-value bound.
    It covers speed uncertainty and numeric/model/reference-map errors.
    Direct headings must snap even at referenceTAS+speed_uncertainty_mps; their
    exact delivery and an appropriate speed-error bound remain assumptions.

    Both endpoint balls must fit the SAME finite-segment capsule in every tick.
    Convexity then covers the full linear interpolation. A failed corner span is
    rejected conservatively; no safety conclusion follows from sampled distances.
    scoring_roundoff_m is another external hypothesis bounding numerical scoring
    error; zero means the ideal real-arithmetic scoring metric is assumed.
    """
    states, commands = tuple(reference_states), tuple(commands)
    if (not 1 <= len(commands) <= MAXIMUM_STEPS or len(states) != len(commands)+1
            or any(not isinstance(value, KinematicState) for value in states)
            or any(not isinstance(value, HeadingCommand) for value in commands)):
        raise ValueError('Expected finite initial+poststep states and matching HeadingCommands')
    dt, accel, bank_deg, earth_radius_m, gravity_mps2, minimum_turn_tas_mps = _parameters(
        dt, accel, bank_deg, earth_radius_m, gravity_mps2, minimum_turn_tas_mps)
    width = Fraction(_number(half_width_m, 'half_width_m', positive=True))
    scoring_error = Fraction(_number(scoring_roundoff_m, 'scoring_roundoff_m', nonnegative=True))
    speed_margin = _number(speed_uncertainty_mps, 'speed_uncertainty_mps', nonnegative=True)
    origin = _latlon(origin_latlon)
    route = tuple(_latlon(point) for point in nominal_latlon)
    if len(route) < 2 or len(latitude_band_deg) != 2:
        raise ValueError('Expected a nominal route and an independently supplied latitude band')
    band = tuple(_number(value, 'latitude band') for value in latitude_band_deg)
    if not -89 < band[0] <= band[1] < 89:
        raise ValueError('Predeclared latitude band must lie strictly within(-89,89)')
    if any(abs(point[1]-origin[1]) >= 180 for point in (*route, *(_latlon(state) for state in states))):
        raise ValueError('This certificate excludes longitude wrapping')
    if isinstance(initial_error_xy_m, (int, float)):
        initial_error_xy_m = (initial_error_xy_m, initial_error_xy_m)
    if len(initial_error_xy_m) != 2:
        raise ValueError('Initial error needs two nonnegative local-axis bounds')
    ex, ey = (Fraction(_number(value, 'initial axis error', nonnegative=True)) for value in initial_error_xy_m)
    if isinstance(per_step_defect_m, (int, float)):
        defects = (Fraction(_number(per_step_defect_m, 'per-step defect', nonnegative=True)),)*len(commands)
    else:
        defects = tuple(Fraction(_number(value, 'per-step defect', nonnegative=True)) for value in per_step_defect_m)
        if len(defects) != len(commands):
            raise ValueError('One declared defect bound is required per tick')
    if not isinstance(initial_error_justification, str) or not initial_error_justification.strip() or not isinstance(defect_justification, str) or not defect_justification.strip():
        raise ValueError('Record the external error-bound hypotheses explicitly')
    scale = Fraction(math.pi*PROJECT_RADIUS_M/180.)
    cos_origin = Fraction(math.cos(math.radians(origin[0])))
    def xy(point):
        latitude, longitude = _latlon(point)
        return ((Fraction(longitude)-Fraction(origin[1]))*scale*cos_origin,
                (Fraction(latitude)-Fraction(origin[0]))*scale)
    vertices, positions = tuple(map(xy, route)), tuple(map(xy, states))
    for a, b in zip(vertices, vertices[1:]):
        _distance_squared(a, a, b)
    nearest_equator = 0. if band[0] <= 0 <= band[1] else min(abs(value) for value in band)
    max_abs_latitude = max(abs(value) for value in band)
    cos_lower, _ = _cos_bounds(max_abs_latitude)
    if cos_lower <= 0:
        raise ValueError('Latitude-band cosine lower bound is not positive')
    sin_upper = _cos_bounds(Fraction(90)-Fraction(max_abs_latitude))[1]
    sec_tan_upper = sin_upper/(cos_lower*cos_lower)
    mbar = max(Fraction(1), _cos_bounds(nearest_equator)[1]/cos_origin)
    capsule_radius = (width-scoring_error)/mbar
    failures, errors, intervals = [], [], []
    max_command_ratio = max_actual_ratio = 0.
    scalar_matches = True
    bank_sin_lower = _cos_bounds(Fraction(90)-Fraction(bank_deg))[0]
    bank_cos_upper = _cos_bounds(bank_deg)[1]
    turn_factor_lower = Fraction(dt)*Fraction(gravity_mps2)*bank_sin_lower/bank_cos_upper*180/PI_HI
    for index, state in enumerate(states):
        if index:
            before, command, eta = states[index-1], commands[index-1], defects[index-1]
            predicted = native_step(before.lat_deg, before.lon_deg, before.hdg_deg, before.tas_mps,
                command.hdg_deg, command.tas_mps, dt=dt, accel=accel, bank_deg=bank_deg,
                earth_radius_m=earth_radius_m, gravity_mps2=gravity_mps2,
                minimum_turn_tas_mps=minimum_turn_tas_mps)
            if predicted != state:
                scalar_matches = False
                failures.append(dict(kind='reference_differs_from_scalar_command_map', tick=index))
            limit = turn_factor_lower/max(Fraction(state.tas_mps)+Fraction(speed_margin), Fraction(minimum_turn_tas_mps))
            requested = abs((Fraction(command.hdg_deg)-Fraction(before.hdg_deg)+180) % 360-180)
            actual = abs((Fraction(state.hdg_deg)-Fraction(before.hdg_deg)+180) % 360-180)
            max_command_ratio = max(max_command_ratio, _ceil_float(requested/limit))
            max_actual_ratio = max(max_actual_ratio, _ceil_float(actual/limit))
            if requested > limit or state.hdg_deg != command.hdg_deg % 360.:
                failures.append(dict(kind='direct_heading_not_certified_to_snap', tick=index,
                    requested_turn_deg=float(requested), speed_margin_turn_limit_deg=float(limit)))
            ey = _grid_upper(ey+eta)
            ex = _grid_upper(ex+Fraction(dt)*Fraction(state.tas_mps)*_sin_upper(state.hdg_deg)*cos_origin/Fraction(earth_radius_m)*sec_tan_upper*ey+eta)
        radius = _sqrt_upper(ex*ex+ey*ey)
        lower_lat, upper_lat = Fraction(state.lat_deg)-ey/scale, Fraction(state.lat_deg)+ey/scale
        if lower_lat < Fraction(band[0]) or upper_lat > Fraction(band[1]):
            failures.append(dict(kind='error_tube_leaves_predeclared_latitude_band', tick=index))
        errors.append((ex, ey, radius))
    for index, (start, end) in enumerate(zip(positions, positions[1:])):
        left_radius, right_radius = errors[index][2], errors[index+1][2]
        chosen = None
        for segment, (a, b) in enumerate(zip(vertices, vertices[1:])):
            if (capsule_radius >= left_radius and capsule_radius >= right_radius
                    and _distance_squared(start, a, b) <= (capsule_radius-left_radius)**2
                    and _distance_squared(end, a, b) <= (capsule_radius-right_radius)**2):
                chosen = segment
                break
        if chosen is None:
            failures.append(dict(kind='no_single_finite_capsule_contains_both_endpoint_balls', interval=index))
        intervals.append(dict(interval=index, nominal_segment_index=chosen,
                              conditional_capsule_check_passed=chosen is not None))
    raw = [raw_centerline_distance_m(state, route) for state in states]
    raw_outside = [index for index, distance in enumerate(raw) if distance > float(width)]
    if raw_outside:
        failures.append(dict(kind='raw_reference_samples_outside_original_metric', ticks=raw_outside))
    return dict(schema='bluesky.experimental-lateral-certificate.v1',
        conditional_certificate=not failures, failures=failures,
        finite_interpolation_only=True, native_continuous_safety_established=False,
        hypotheses_verified_by_module=False,
        assumptions=dict(initial_position_error=initial_error_justification, local_position_defect=defect_justification,
            scoring_roundoff_m=float(scoring_error),
            scoring_roundoff_scope='Externally supplied absolute upper bound; zero assumes ideal real-arithmetic scoring.',
            direct_heading='Exact direct-heading delivery and initially matching heading; the supplied TAS uncertainty bound must hold on every tick.',
            interpolation='Actual local positions are linearly interpolated between ticks. Endpoint errors alone do not bound an unspecified within-tick path.',
            model='No wind/ASAS/turnphi override; resolved TAS intent; each supplied axis defect bounds the sum of actual and stored-reference residuals against the prescribed-reference-V/heading real position map, uniformly on admissible states.'),
        parameters=dict(dt=dt, accel=accel, bank_deg=bank_deg, earth_radius_m=earth_radius_m,
            gravity_mps2=gravity_mps2, minimum_turn_tas_mps=minimum_turn_tas_mps,
            origin_latlon=list(origin), local_radius_m=PROJECT_RADIUS_M, latitude_band_deg=list(band),
            half_width_m=float(width), speed_uncertainty_mps=speed_margin),
        projection_mbar_upper=_ceil_float(mbar), sec_tan_sup_upper=_ceil_float(sec_tan_upper),
        certificate_capsule_radius_m=float(capsule_radius),
        geometry_arithmetic='Exact rational projection using the fixed project scale/cos-origin floats as declared coefficients; rational squared-distance inequalities, outward1e-12m axis-error rounding at every tick, and outward1e-12m radius bounds. This does not establish supplied simulator/scoring error hypotheses.',
        reference_scalar_map_exact=scalar_matches,
        max_command_heading_increment_rate_ratio=max_command_ratio,
        max_actual_heading_increment_rate_ratio=max_actual_ratio,
        error_bounds=[dict(tick=index, ex_m=_ceil_float(x), ey_m=_ceil_float(y), radius_m=_ceil_float(radius))
                      for index, (x, y, radius) in enumerate(errors)],
        intervals=intervals,
        raw_reference_metric=dict(distances_m=raw, max_distance_m=max(raw), outside_sample_indices=raw_outside,
            strict_comparison='distance > original half-width; no epsilon or sample deletion', constitutes_certificate=False))
