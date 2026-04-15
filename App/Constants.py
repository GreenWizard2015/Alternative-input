"""Constants for Application modes and UI components.

This module centralizes all magic numbers and configuration parameters used
across the App package to enable easier tuning and debugging.
"""

# MoveToGoal mode constants
MOVE_TO_GOAL_BASE_SPEED = 55  # pixels per second base speed
MOVE_TO_GOAL_SPEED_MULTIPLIER = 4  # Total multiplier: 55 * 4 = 220 px/s
MOVE_TO_GOAL_DISTANCE_THRESHOLD = 3.0  # pixels - threshold to pick new goal

# CircleMovingMode constants
CIRCLE_MODE_TRANSITION_TIME = 2.0  # seconds between goal transitions
CIRCLE_MODE_MAX_LEVEL = 25  # maximum difficulty level
CIRCLE_MODE_PATH_OFFSET = 0.5  # center offset for circular path
CIRCLE_MODE_DIFFICULTY_DIVISOR = 2.0  # divisor for level calculation

# LookAtMode constants
LOOK_AT_MODE_VISIBLE_TIME = 5.0  # seconds target stays visible before relocating

# GameMode constants
GAME_MODE_RADIUS_DELTA_PER_SECOND = 0.01  # change in target radius per second
GAME_MODE_MAX_HITS = 5  # maximum hits for game mode
GAME_MODE_MAX_HITS_IN_RANGE = 5  # maximum hits for targets in range
GAME_MODE_TARGET_SCALE = 2  # scale factor for target visualization
GAME_MODE_ADJUST_POS_THRESHOLD = 0.5  # threshold for position adjustment in _adjustPos
GAME_MODE_ADJUST_POS_POWER = (
    4  # default power exponent for edge concentration in _adjustPos
)

# Camera/Display constants
CAMERA_WIDTH = 1024  # default camera capture width
CAMERA_HEIGHT = 768  # default camera capture height
CAMERA_AUTO_EXPOSURE = -5  # camera auto-exposure setting
CAMERA_AUTOFOCUS = 0  # camera autofocus setting (disabled)
CAMERA_AUTO_WB = 0  # camera auto white balance setting (disabled)

# Timing constants

# Background animation constants
BACKGROUND_BRIGHTNESS_CYCLE_DURATION = 30.0  # seconds for one brightness cycle
BACKGROUND_BRIGHTNESS_AMPLITUDE = 1.0 / 2.0  # brightness amplitude (0.5)
BACKGROUND_COLOR_CYCLE_INTERVAL = 5  # seconds between background color changes

# SpinningTarget constants
SPINNING_TARGET_SATELLITE_COUNT = 5  # number of satellite objects
SPINNING_TARGET_TIME_SCALE = 10  # time scale for radius animation
SPINNING_TARGET_RADIUS_AMPLITUDE = 0.015  # maximum radius amplitude for oscillation
SPINNING_TARGET_ROTATION_SPEED = 0.1  # rotation speed in radians per frame
SPINNING_TARGET_INITIAL_ANGLE_MAX = (
    2.0 * 3.141592653589793
)  # full circle range (2π radians)
SPINNING_TARGET_INITIAL_POSITION_CENTER = (
    0.5  # initial position at center of normalized coordinates
)

# SplineMode constants
SPLINE_MODE_OVERLAP_POINTS = 3  # number of spline control points for overlap

# IlluminationSource constants
ILLUMINATION_SPLINE_OVERLAP_POINTS = 3  # number of spline control points
ILLUMINATION_SPEED_MIN = 1.0  # minimum speed for light source
ILLUMINATION_SPEED_MAX = 4.0  # maximum speed for light source
ILLUMINATION_DURATION_MIN = 20  # minimum duration in time units
ILLUMINATION_DURATION_MAX = 40  # maximum duration in time units
ILLUMINATION_RADIUS = 310  # rendering radius for light source
ILLUMINATION_SPLINE_CONTROL_POINTS = 4  # number of control points for spline
ILLUMINATION_POSITION_CENTER = 0.5  # center position for light source
ILLUMINATION_POSITION_SCALE = 0.5  # scale for position randomization
ILLUMINATION_CLIP_MIN = -0.5  # minimum clip value for spline points
ILLUMINATION_CLIP_MAX = 1.5  # maximum clip value for spline points
ILLUMINATION_RENDER_RADIUS_SCALE = 0.1  # scale factor for rendering radius

# GameMode constants (additional)
GAME_MODE_MAX_PROB_POWER = 10  # maximum probability power for target distribution
GAME_MODE_MAX_IN_RANGE_HITS = 25  # maximum hits for in-range targets
GAME_MODE_INITIAL_POSITION = 0.5  # initial center position for game mode (0.0-1.0)
GAME_MODE_POSITION_CLIP_MIN = 0.0  # minimum clip value for positions
GAME_MODE_POSITION_CLIP_MAX = 1.0  # maximum clip value for positions
GAME_MODE_RENDER_CIRCLE_RADIUS = (
    25  # fixed radius for hit detection circle rendering (pixels)
)

# CornerMode constants
CORNER_MODE_RADIUS = 0.05
CORNER_MODE_FREQUENCY_MULTIPLIER = 4
CORNER_MODE_CLIP_MIN = 0.0
CORNER_MODE_CLIP_MAX = 1.0

# SplineMode generation and movement constants
SPLINE_SCALE_MIN = 0.1
SPLINE_SCALE_MAX = 0.2
SPLINE_POINT_OFFSET = 0.5
SPLINE_NORMALIZATION_EPSILON = 1e-6
SPLINE_POINT_CLIPPING_MIN = -0.5
SPLINE_POINT_CLIPPING_MAX = 1.5
SPLINE_SPEED_MIN = 0.15
SPLINE_SPEED_MAX = 1.0
SPLINE_DURATION_MIN_MULTIPLIER = 3
SPLINE_DURATION_MAX_MULTIPLIER = 10
SPLINE_POSITION_CLIPPING_MIN = 0.0
SPLINE_POSITION_CLIPPING_MAX = 1.0
