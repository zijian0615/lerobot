"""
Scene parameters for the FANUC ER-4iA grasping cell.

Coordinate frame: origin at the robot base mounting face (on the table top),
+X pointing away from the robot toward the work area / camera side,
+Y to the robot's left, +Z up. Meters and degrees, matching the asset
(metersPerUnit = 1, upAxis = Z).

Provenance tags:
  EXACT     - read out of the NVIDIA/FANUC SimReady USD, or a vendor datasheet
  SCAN      - measured off the Scaniverse site scan (CSI-site-digital-model.zip,
              "204107.glb"), registered to the robot by its base-plate bolts
  VIDEO     - measured in the real-cell video (pick_m6_scrwes_CSI_hackathon26.mov),
              mapped onto the scan by a homography through printed mat features
              (~2 mm RMS), or scaled against a part of known size
  PHOTO     - estimated from the 2026-09 bench photos of the gripper
  GUESS     - not observable; placeholder pending measurement
"""

# ------------------------------------------------------------------ robot
# EXACT. Present only in the Isaac 6.0 asset bundle; 5.0/5.1 ship CRX10IAL only.
ROBOT_USD_SUBPATH = "/Isaac/Robots/Fanuc/er4ia/er4ia.usd"
ROBOT_PRIM        = "/World/ER4iA"

# EXACT (er4ia_physics.usd). Degrees.
JOINT_LIMITS_DEG = {
    "J1": (-170.0, 170.0), "J2": (-110.0, 120.0), "J3": (-69.0, 205.0),
    "J4": (-190.0, 190.0), "J5": (-120.0, 120.0), "J6": (-360.0, 360.0),
}

# EXACT (joint localPos chain). Useful for sanity-checking the layout scale.
SHOULDER_HEIGHT = 0.33    # base mount face -> J2 axis
UPPER_ARM       = 0.26    # J2 -> J3
ELBOW_OFFSET    = 0.02    # J3 -> J4
FOREARM         = 0.29    # J4 -> J5 (wrist center)
MAX_REACH       = 0.55    # ~ UPPER_ARM + FOREARM

# VIDEO: arm hangs forward and down over the work area between moves.
HOME_POSE_DEG = {"J1": 0.0, "J2": 35.0, "J3": -25.0, "J4": 0.0, "J5": -80.0, "J6": 0.0}

# Where the arm parks while the overhead camera takes the planning snapshot:
# swung ~80 deg to the robot's right, fingertips 0.30 m above the mat at
# (0.05, -0.32) -- clear of the screws and the bin. (At the zero pose the
# fingers hang straight over the screw target and hide one from the camera.)
CAPTURE_POSE_DEG = {"J1": -81.1, "J2": 7.4, "J3": 5.8, "J4": 0.0, "J5": -88.4, "J6": 55.9}

# VIDEO: the real J1 base casting is mid grey, not black as the asset renders it.
ROBOT_BASE_GREY = True

# ------------------------------------------------------------------ work cell
# GUESS. The site scan does not reach the floor; it only shows the top is at
# least 687 mm up. Only affects world height, not the task.
TABLE_TOP_Z  = 0.90
# SCAN: tabletop extents in the robot frame. The robot is square to the table
# (base-plate and table edges agree to within 0.1 deg) and sits 371 mm from the
# back edge, not centred.
TABLE_X      = (-0.371, 0.759)
TABLE_Y      = (-0.772, 0.792)
TABLE_THICK  = 0.05
TABLE_SIZE   = (TABLE_X[1] - TABLE_X[0], TABLE_Y[1] - TABLE_Y[0], TABLE_THICK)
TABLE_CENTER = ((TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2)
# SCAN + VIDEO: yellow "CAUTION" tape along the front and both side edges of
# the tabletop (not the back edge, which abuts the enclosure).
EDGE_BAND_W  = 0.048

# SCAN: the printed mat artwork, rectified into the robot frame by
# make_mat_texture.py. It is a stock FANUC LR Mate mat -- its arcs are labelled
# "200iD" / "200iD/4S" and are centred ~0.37 m BEHIND the ER-4iA's J1, which is
# why the earlier assumption (arcs = ER-4iA reach, centred on J1) was wrong.
# The "#" of black tape next to the bin and the black pad are in it too.
MAT_TEXTURE  = "assets/mat_print.png"
MAT_MARKINGS = True       # False -> plain white laminate

# ------------------------------------------------------------------ props
# VIDEO: dark fixture at the table's front-left corner (+X, -Y): a grey base
# plate carrying a raised tray, a two-hole block and a pill bottle. It is a
# distractor for perception, not a task object. Footprint mapped from the
# video; internal layout approximate.
FIXTURE_XY      = (0.555, -0.640)
FIXTURE_PLATE   = (0.230, 0.260, 0.012)
FIXTURE_TRAY    = (0.130, 0.170, 0.026)     # raised tray, recessed top
FIXTURE_BLOCK   = (0.055, 0.115, 0.034)     # block with two round holes
BOTTLE_R        = 0.018
BOTTLE_H        = 0.058

# ------------------------------------------------------------------ gripper
# EOAT, flange -> fingertip. Modelled from the 2026-09 bench photos and the
# real-cell video. ABSOLUTE SCALE now comes from the video: the Banner
# LEDRR70X70-78587 ring light on the wrist is 76.2 x 76.2 x 29.2 mm per its
# datasheet, and against it the flange-to-fingertip length measures ~300 mm
# (+/- 7%) with the Z cylinder extended for a pick. The earlier 290 mm guess was
# close; the stack proportions were not -- the real jaw assembly is ~32 mm
# across, not ~90 mm.
#
# Stack (VIDEO): black adapter -> round silver pneumatic Z cylinder carrying a
# white clamp collar (it holds the camera bracket) -> narrow rod guide -> small
# pneumatic parallel gripper with push-in fittings -> two taped blade fingers.
# A cylindrical Class 2 laser pointer rides alongside the fingers (not modelled;
# its red dot is visible on the mat in the video).
TOOL_LENGTH       = 0.300     # VIDEO, flange face to fingertip, +/- 0.02
_s                = TOOL_LENGTH / 0.300      # scale factor on the figures below

ADAPTER_R         = 0.030 * _s   # black, ~60 mm across
ADAPTER_T         = 0.044 * _s
ZCYL_R            = 0.0235 * _s  # ~47 mm silver body
ZCYL_LEN          = 0.088 * _s
COLLAR_R          = 0.032 * _s   # white clamp collar round the cylinder
COLLAR_T          = 0.028 * _s
COLLAR_AT         = 0.018 * _s   # down the cylinder from its top
ZFLANGE_R         = 0.016 * _s   # narrow rod guide under the cylinder
ZFLANGE_T         = 0.020 * _s
BODY_LEN          = 0.030 * _s   # compact parallel gripper body
BODY_W            = 0.030 * _s
BODY_D            = 0.026 * _s
PLATE_T           = 0.006 * _s   # jaw guide
PLATE_W           = 0.036 * _s
PLATE_D           = 0.022 * _s
CARRIAGE_LEN      = 0.010 * _s   # jaw carriages the fingers bolt to
CARRIAGE_W        = 0.010 * _s
CARRIAGE_D        = 0.018 * _s
CARRIAGE_OVERLAP  = 0.007 * _s   # finger root is clamped inside the carriage

# The fingers take whatever is left of TOOL_LENGTH, so the fingertip is exactly
# TOOL_LENGTH from the flange.
FINGER_LENGTH     = TOOL_LENGTH - (ADAPTER_T + ZCYL_LEN + ZFLANGE_T + BODY_LEN
                                   + PLATE_T + CARRIAGE_LEN - CARRIAGE_OVERLAP)
FINGER_WIDTH      = 0.010 * _s   # VIDEO: along the closing direction
FINGER_THICK      = 0.012 * _s
# PHOTO: a semicircular scallop on the inner face near the tip. Covered by the
# tape and too high to reach anything lying on the table, so it is cosmetic.
# Must satisfy SCALLOP_AT + SCALLOP_R/FINGER_LENGTH < 1 or the arc runs past the
# tip and the profile self-intersects into a garbage collider -- which silently
# breaks every grasp.
SCALLOP_R         = 0.0035 * _s
SCALLOP_AT        = 0.86

# Per-jaw travel; inner faces sit at +/- these from the tool axis. Stroke is
# still UNMEASURED. Closed: a pneumatic gripper is commanded fully shut and
# stalls on the part, so this only needs to be narrower than the thinnest thing
# gripped (the 6 mm shank). Open: clears a 12 mm countersunk head with margin
# while keeping the jaws about as wide as the video shows.
JAW_CLOSED        = 0.002
JAW_OPEN          = 0.009
# How far above the work surface the fingertips are commanded when picking.
# At 1 mm the position-controlled arm pressed the taped tips into the mat, and
# tape-on-mat friction held the jaws open: they were still 1.5-5 mm short of
# the part when the lift began, and one screw was left behind. 2.5 mm keeps the
# tips off the mat and still below a lying screw's axis (~4.5 mm).
TIP_CLEARANCE     = 0.0025

# Duct tape wrapped round the working end of each blade, as on the real
# gripper. It raises the friction of the flat-face pinch.
TAPE_LEN          = 0.032       # how far up from the tip it is wrapped
TAPE_PAD          = 0.0006      # how proud of the blade it sits
TAPE_STATIC_FRIC  = 1.5
TAPE_DYNAMIC_FRIC = 1.3
JAW_MASS          = 0.025       # one blade + its carriage
JAW_DRIVE_STIFF   = 4.0e4
JAW_DRIVE_DAMP    = 3.0e3
JAW_MAX_FORCE     = 180.0

TOOL_AXIS          = (1.0, 0.0, 0.0)
FINGER_SPREAD_AXIS = (0.0, 1.0, 0.0)

# ------------------------------------------------------------------ wrist camera
# VIDEO: a FANUC SC130EF2 (Kowa-made, 1/1.8" 1.3 MP mono) sits ON TOP of a
# Banner LEDRR70X70-78587 red LED ring light and looks straight down through it,
# parallel to the tool, bracketed off the white collar. The ring light is the
# biggest thing on the wrist and was missing from the model entirely.
RINGLIGHT         = (0.0762, 0.0762, 0.0292)   # EXACT, Banner datasheet
RINGLIGHT_WINDOW_R = 0.0192                    # EXACT, 38.3 mm window
CAMBODY           = (0.029, 0.029, 0.030)      # XC-56 form factor (SC130EF2)
# Side photos real_eoat_new_side.jpeg, scaled by the Gemini 305 housing
# (42 mm long, 23 mm thick). The lens sits on the black barrel at the inner
# end of the body. Cylinder axis to lens ~84 mm; lens centre ~150 mm above
# the taped fingertip. ±15 mm: the phone is not perfectly square to the tool.
# Look direction stays parallel to the tool. The old 65 mm was the ring light,
# which hung much lower.
WRISTCAM_LATERAL  = 0.084
WRISTCAM_DIR      = (0.0, 0.0, 1.0)            # flange-frame side it hangs on
RINGLIGHT_ABOVE_TIP = 0.150
BRACKET_T         = 0.006

# ------------------------------------------------------------------ task objects
# The real cell's job ("pick all screws and put them into the yellow
# container", per its operator UI) is screws only. The coloured cubes were
# added to the sim on request; set INCLUDE_CUBES to bring them back.
INCLUDE_CUBES     = False
CUBE_EDGE         = 0.020
CUBE_MASS         = 0.0096      # 20 mm moulded plastic, ~1200 kg/m3
CUBE_ROW_X        = 0.270       # clear of the screw target and the bin
CUBE_ROW_Y        = (-0.100, -0.040, 0.020) if INCLUDE_CUBES else ()
CUBE_MATS         = ["plastic_red", "plastic_yellow", "plastic_blue"]
if INCLUDE_CUBES:
    JAW_OPEN = 0.012            # a 20 mm cube needs more than the screw opening

# Objects rest on the printed mat, which sits 2 mm proud of the table top.
# Grasp heights and the back-projection plane must both use this, or every
# grasp is 2 mm low.
SURFACE_DZ        = 0.002

# VIDEO: M6 x 30 COUNTERSUNK socket screws (ISO 10642), black oxide -- the head
# is a 90 deg cone, clearly visible side-on in the footage. (An earlier pass
# modelled cylindrical socket-head caps; that was a misreading.)
SCREW_LENGTH      = 0.0300      # overall, head included, as for any countersunk
SCREW_HEAD_D      = 0.0120
SCREW_HEAD_H      = 0.0033
SCREW_SHANK_D     = 0.0060
SCREW_MASS        = 0.0076      # steel, ~0.97 cm3
# VIDEO: the eight screws' starting layout at t=10 s, clustered inside the
# printed screw target. (x, y) is the HEAD, yaw is the head->tip direction.
SCREWS            = [(0.435, -0.128,   24.5),
                     (0.420, -0.097,   98.5),
                     (0.401, -0.054,  150.3),
                     (0.339, -0.079,    3.7),
                     (0.427, -0.016, -160.9),
                     (0.371,  0.014,   32.7),
                     (0.438,  0.025,  -62.5),
                     (0.433,  0.088,  -92.5)]

# ------------------------------------------------------------------ bin
# VIDEO: the collection bin is "BLOCK-03", a yellow 3D-printed cube with a
# thick wall and a rounded-square pocket. It is NOT the scanned orange
# "BLOCK-06" in assets/ (thin-walled, never seen in the video). Built
# procedurally; size is scaled off the screws and the mat in the video.
BIN_XY            = (0.370, 0.161)
BIN_OUTER         = (0.060, 0.060)
BIN_WALL_H        = 0.060
BIN_POCKET        = 0.036       # square pocket opening
BIN_POCKET_R      = 0.008       # its corner radius
BIN_FLOOR_T       = 0.015       # GUESS: pocket depth is not visible
BIN_WALL_T        = (BIN_OUTER[0] - BIN_POCKET) / 2
BIN_SIZE          = (BIN_POCKET, BIN_POCKET)   # usable interior, for drops
BIN_RELEASE_Z     = 0.092                      # above the rim

# ------------------------------------------------------------------ cameras
# Cameras are specified as eye + look_at in the parent's frame, not Euler
# angles -- it is the only formulation that stays readable while iterating.

# The wrist camera at its REAL mount: the SC130EF2 lens, looking straight down
# the tool direction through the ring light. iRVision is used camera-over-part
# -- the arm parks the camera above the scene, snaps, then moves the gripper --
# so this view does not contain the fingers, by design.
# 2 mm below the ring light's window so the housing cannot occlude the view
_WC_AT  = TOOL_LENGTH - RINGLIGHT_ABOVE_TIP + 0.002
_WC_EYE = tuple(a * _WC_AT + d * WRISTCAM_LATERAL
                for a, d in zip(TOOL_AXIS, WRISTCAM_DIR))
# Intrinsics: Orbbec Gemini 305 color, datasheet typical (not this unit).
# 1280x720 is the 16:9 crop of the 1280x800 sensor: HFOV 94°, VFOV 62°, ±3°.
# Distortion < 1.5%, so a pinhole is enough. fx = 640/tan(47°) = 597 px.
# Mount is the side-photo estimate above: 84 mm off the tool axis, 150 mm
# above the fingertips, looking straight down the tool.
WRIST_CAM = dict(
    parent         = ROBOT_PRIM + "/J6_link",
    eye            = _WC_EYE,
    look_at        = tuple(e + a * 0.2 for e, a in zip(_WC_EYE, TOOL_AXIS)),
    up             = (0.0, 1.0, 0.0),          # flange coordinates
    resolution     = (1280, 720),
    focal_length   = 9.7704,                   # fx 597 px at horiz_aperture 20.955
    horiz_aperture = 20.955,
)

# Current overhead: Orbbec Gemini color, 1920x1080. Intrinsics are the
# undistorted pinhole fit on that stream (fx = fy = 1249 px, principal point
# at the centre). horiz_aperture 20.955 mm is the USD aperture that pair uses,
# so focal_length = fx * aperture / width.
# Pose is that same solve, in this robot frame: the calibration's XY is the
# controller base (J1 axis, same origin as the mounting face) and z = 0 is the
# table plane the samples were touched on. Eye is 0.765 m forward, 0.049 m to
# the robot's left, 0.929 m above the table, tilted 26.7 deg off nadir. The
# optical axis meets the table at (0.297, 0.052).
TABLE_CAM = dict(
    parent         = "/World",
    eye            = (0.7646, 0.0486, TABLE_TOP_Z + 0.9289),
    look_at        = (0.2966, 0.0521, TABLE_TOP_Z),
    up             = (-0.8930, 0.0043, 0.4500),
    resolution     = (1920, 1080),
    focal_length   = 13.633273452911222,
    horiz_aperture = 20.955,
    fisheye        = False,
)

# Fixed side webcam: Shenzhen AONI Full HD, 1920x1080, reference/side_aoni_1920.jpg.
# Plane solve against the orthophoto (same J1 registration as the overhead).
# Correspondences: the black "#" tape, the BATTERY KIT lettering, and the
# diamond in the nested-square target. Principal point at the image centre,
# no distortion. Reprojection RMS about 13 px, so the pose is good to roughly
# a centimetre, not to the overhead's sample fit. The lens sits on the robot's
# left, about 0.12 m above the table, looking across the mat toward -Y with
# the optical axis a few degrees above the horizon (the frame is mostly wall
# and arm, table in the lower half).
SIDE_CAM = dict(
    parent         = "/World",
    eye            = (0.4895, 0.5050, TABLE_TOP_Z + 0.1232),
    look_at        = (0.3735, -0.0302, TABLE_TOP_Z + 0.1727),
    up             = (0.0930, 0.0721, 0.9931),
    resolution     = (1920, 1080),
    focal_length   = 8.382,          # 768 px at horiz_aperture 20.955; HFOV ~103 deg
    horiz_aperture = 20.955,
    fisheye        = False,
)

# A tighter third view on the work area. Not a real camera in the cell -- it is
# there to give the localisation work a close, well-lit look at the objects.
CLOSEUP_CAM = dict(
    parent         = "/World",
    eye            = (0.80, -0.26, TABLE_TOP_Z + 0.36),
    look_at        = (0.39,  0.03,  TABLE_TOP_Z + 0.01),
    up             = (0.0,   0.0,  1.0),
    resolution     = (1280, 960),
    focal_length   = 18.0,
    horiz_aperture = 20.955,
)

# VIDEO: both cameras frame this region.
WORK_AREA_CENTER = (0.40, 0.0, 0.0)
