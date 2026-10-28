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
# REAL (2026-10-08): the asset paints the arm pure (1, 1, 0), which renders lime. The real arm reads
# orange-yellow, R > G (table cam (116, 99, 9), side cam (150, 119, 69) against the asset's
# (188, 187, 25) / (169, 165, 17)); fitted in linear light (sRGB ratio ** 2.2), averaged over both cameras.
ROBOT_YELLOW = (0.45, 0.28, 0.03)

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
MAT_TEXTURE  = "assets/surface_real.png"  # 2026-10-08 white surface, make_real_mat.py --scan; old: assets/mat_print.png
MAT_MARKINGS = True       # False -> plain white laminate

# ------------------------------------------------------------------ props
# VIDEO: dark fixture at the table's front-left corner (+X, -Y): a grey base
# plate carrying a raised tray, a two-hole block and a pill bottle. It is a
# distractor for perception, not a task object. Footprint mapped from the
# video; internal layout approximate.
# REAL (2026-10-07): not in the current cell, so it is left out.
INCLUDE_FIXTURE = False
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
#
# REAL (2026-10-07), overrides the video scale. The gripper hangs off the
# J6_link origin, which is the WRIST CENTRE (J5 axis), not the faceplate. On
# the controller, UTOOL1 = (0, 0, 223) mm off the faceplate (FRC_ReadUToolData)
# and the faceplate is 70 mm out from the wrist centre (four FRC poses, see
# check_j3_convention.py), so the fingertip is 293 mm from this origin. This
# takes UTOOL1 to be taught at the fingertip.
TOOL_LENGTH       = 0.293     # REAL, J6_link origin (wrist centre) to fingertip
_s                = TOOL_LENGTH / 0.300      # scale factor on the figures below (drawn at 300)

ADAPTER_R         = 0.030 * _s   # black, ~60 mm across
ADAPTER_T         = 0.044 * _s
ZCYL_R            = 0.0235 * _s  # ~47 mm silver body
# ZCYL_LEN is set below FINGER_LENGTH: it takes up whatever the measured parts leave of TOOL_LENGTH.
COLLAR_R          = 0.032 * _s   # white clamp collar round the cylinder
COLLAR_T          = 0.028 * _s
COLLAR_AT         = 0.018 * _s   # down the cylinder from its top
ZFLANGE_R         = 0.016 * _s   # narrow rod guide under the cylinder
ZFLANGE_T         = 0.020 * _s
# REAL (2026-10-09): SMC MHZL2-16D2 (long-stroke parallel gripper, 16 mm bore, through-hole fingers).
# Lengths along the tool from the 2026-10-09 photos, scaled so the flange-to-jaw-bottom stack is
# 223 - 76 = 147 mm: Z cylinder + plate ~90, gripper body + guide ~45, SMC jaw ~12.
BODY_LEN          = 0.039        # gripper body (photo, scaled)
BODY_W            = 0.030 * _s
BODY_D            = 0.026 * _s
PLATE_T           = 0.006        # jaw guide
PLATE_W           = 0.036 * _s
PLATE_D           = 0.022 * _s
CARRIAGE_LEN      = 0.0188       # the SMC jaws the blades bolt to: 12 mm shows below the guide
CARRIAGE_W        = 0.010 * _s
CARRIAGE_D        = 0.018 * _s
CARRIAGE_OVERLAP  = 0.0068       # finger root is clamped inside the carriage

# REAL (2026-10-09): the blade shows 76 mm below the SMC jaw (measured). It was 106.5 mm when the fingers
# took whatever the estimated stack left of TOOL_LENGTH. Now the Z cylinder takes the slack, so the
# fingertip stays exactly TOOL_LENGTH from the J6 origin (wrist centre; the J6 faceplate is 70 mm out).
FINGER_LENGTH     = 0.076
ZCYL_LEN          = TOOL_LENGTH - (ADAPTER_T + ZFLANGE_T + BODY_LEN + PLATE_T
                                   + CARRIAGE_LEN - CARRIAGE_OVERLAP + FINGER_LENGTH)
FINGER_WIDTH      = 0.010 * _s   # VIDEO: along the closing direction
FINGER_THICK      = 0.012 * _s
# PHOTO: a semicircular scallop on the inner face near the tip. Covered by the
# tape and too high to reach anything lying on the table, so it is cosmetic.
# Must satisfy SCALLOP_AT + SCALLOP_R/FINGER_LENGTH < 1 or the arc runs past the
# tip and the profile self-intersects into a garbage collider -- which silently
# breaks every grasp.
SCALLOP_R         = 0.0035 * _s
SCALLOP_AT        = 0.86

# Per-jaw travel; inner faces sit at +/- these from the tool axis. REAL (2026-10-09): MHZL2-16D strokes
# 12 mm across both jaws (SMC catalogue), 6 mm each. Closed: a pneumatic gripper is commanded fully shut
# and stalls on the part, so this only needs to be narrower than the thinnest thing gripped (the 6 mm
# shank). Open = closed + 6 mm: 16 mm between the blades, clearing a 12 mm countersunk head. (The
# open blade gap itself is still unmeasured; it was 18 mm with a 7 mm-per-jaw guess.)
JAW_CLOSED        = 0.002
JAW_OPEN          = 0.008
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

# ---- CAD GRIPPER (2026-10-09). Merged from the hackathon starter: SMC MHZL2-16D2 body and jaws from the
# supplied STEP (assets/gripper/smc_mhzl2_16d2.json), blades reconstructed from photos
# (assets/gripper/photo_finger_detail.json, gripper_geometry.py). Overrides the estimated stack above.
# Lengths are the real cell's: the stack starts at the J6 faceplate, FLANGE_OFFSET out from the J6_link
# origin (wrist centre), and faceplate -> fingertip is 223 mm (UTOOL1 Z, ruler), so TOOL_LENGTH = 293 mm.
GRIPPER_MODEL = "SMC MHZL2-16D2"
GRIPPER_CAD = "assets/gripper/smc_mhzl2_16d2.json"
SHOW_TOOL_DETAIL = True
FLANGE_OFFSET = 0.070
ADAPTER_R = 0.026
ADAPTER_T = 0.012
ZCYL_R = 0.023
ZCYL_RECTANGULAR = False
ZCYL_BODY_W = 0.038
ZCYL_BODY_D = 0.034
COLLAR_R = 0.028
COLLAR_T = 0.010
COLLAR_AT = 0.004
ZFLANGE_R = 0.013
ZFLANGE_T = 0.010
BODY_LEN = 0.0527          # CAD body
BODY_W = 0.047
BODY_D = 0.0236
PLATE_T = 0.0              # part of the CAD body
PLATE_W = 0.042
PLATE_D = 0.028
CARRIAGE_LEN = 0.02115
CARRIAGE_W = 0.0155
CARRIAGE_D = 0.008
CARRIAGE_OVERLAP = 0.02115  # the blade starts at the CAD body front
FINGER_LENGTH = 0.076      # REAL: 76 mm below the SMC jaw (measured)
FINGER_WIDTH = 0.012
FINGER_THICK = 0.006
FINGER_TIP_BEVEL = 0.014
FINGER_TIP_WIDTH = 0.0075
FINGER_EDGE_BREAK = 0.0006
FINGER_MOUNT_HOLES = (0.010, 0.023)
FINGER_TAPE = False
AA_DIAMETER = 0.01450
AA_POCKET_CLEARANCE = 0.0
SCALLOP_R = (AA_DIAMETER + AA_POCKET_CLEARANCE) / 2
# The inner hook sets the fingertip gap: 2 * (jaw + sqrt(R^2 - d^2)). d = 6.97 mm gives 4 mm closed and
# 16 mm open, the real cell's 4 +- 1 / 15 +- 2 mm between the blade tips (the starter's 6.5 mm gave 6.4 /
# 18.4 mm, and a 6.4 mm closed gap cannot pinch the 6 mm screw shank).
SCALLOP_ABOVE_TIP = 0.00697
SCALLOP_AT = 1.0 - SCALLOP_ABOVE_TIP / FINGER_LENGTH
FINGER_INNER_TIP = (SCALLOP_R**2 - SCALLOP_ABOVE_TIP**2) ** 0.5
SOFT_TIP_LENGTH = 0.032
FINGER_COLLIDER_T = 0.006   # straight collision blade, inner face at the fingertip gap (visual profile is separate)
SOFT_TIP_STATIC_FRIC = 1.5
SOFT_TIP_DYNAMIC_FRIC = 1.3
JAW_STROKE = 0.006         # SMC catalogue: 12 mm across both jaws
JAW_CLOSED = 0.0           # straight-face half gap; the tips sit FINGER_INNER_TIP further out
JAW_OPEN = JAW_CLOSED + JAW_STROKE
JAW_MASS = 0.032
LASER_LENGTH = 0.032
LASER_RADIUS = 0.0035
CAMBODY = (0.050, 0.050, 0.032)
CAMBODY_GREY = (0.50, 0.50, 0.52)
CAM_CORNER_R = 0.005
CAM_LID_T = 0.002
BRACKET_T = 0.004
WRISTCAM_ROLL_DEG = 90.0
ZCYL_LEN = 0.0723          # takes up the rest of the 223 mm faceplate -> fingertip
TOOL_LENGTH = (FLANGE_OFFSET + ADAPTER_T + ZCYL_LEN + ZFLANGE_T + BODY_LEN + PLATE_T
               + CARRIAGE_LEN - CARRIAGE_OVERLAP + FINGER_LENGTH)
assert abs(TOOL_LENGTH - 0.293) < 1e-6, TOOL_LENGTH
# The wrist camera housing follows the calibrated lens (WRIST_CAM eye below: 0.1633, 0.0125, 0.0971).
WRISTCAM_DIR = (0.0, 0.1277, 0.9918)
WRISTCAM_LATERAL = 0.0979
CAM_ABOVE_TIP = TOOL_LENGTH + 0.006 - 0.1633

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
# REAL (2026-10-07): the solved lens is 97 mm off the tool axis and 163 mm down
# it from the J6 origin (WRIST_CAM below), i.e. 130 mm above the fingertip. The
# housing follows it; the render camera itself uses the solved pose directly.
WRISTCAM_LATERAL  = 0.097
WRISTCAM_DIR      = (0.0, 0.0, 1.0)            # flange-frame side it hangs on
RINGLIGHT_ABOVE_TIP = 0.132
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

# The work surface (what objects rest on), as a height above the ROBOT BASE MOUNTING PLANE (z = 0 of the
# base frame, which sits at TABLE_TOP_Z in the world). Grasp heights and the back-projection plane must
# both use this, or every grasp is off by the error.
# REAL (2026-10-08, white surface): with the arm at home the fingertip measures 245 mm above the surface,
# where FK puts it 239.9 mm above the base plane, so the surface is 5.1 mm BELOW the base plane. (It was
# taken as the old mat, 2 mm proud of a table flush with the base; grasps then stopped ~7 mm high.)
SURFACE_DZ        = -0.0051
MAT_T             = 0.002       # the surface sheet's thickness; only the drawn slab, the table top is below it

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
# REAL (2026-10-08, testbed rebuilt): SCREWS[0], the single-screw task, is where the screw lies now,
# located from the table view (locate_objects.py, data/objects_v2); the video's was (0.435, -0.128, 24.5).
SCREWS            = [(0.416, -0.139,   37.9),
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
# REAL (2026-10-07): fingertip jogged to the pocket centre and read over FRC
# (it moved from the video estimate (0.370, 0.161) by ~10 cm).
# REAL (2026-10-08, testbed rebuilt): a smaller yellow bin, axis-aligned, its pose and outer size fitted to
# the table and side silhouettes (locate_objects.py --bin-height 0.055 on the -5.1 mm surface, IoU 0.87); the 2026-10-07 bin was
# 60 x 60 x 60 mm at (0.3313, 0.2547). The pocket is not measured: scaled with the outer size.
# REAL (2026-10-17): every bin is 50 x 50 x 55 mm, its pocket 35 x 35 mm and 25 mm deep (the organizers'
# measurement; the silhouette fit above gave 47 mm).
BIN_XY            = (0.5110, 0.1076)
BIN_OUTER         = (0.050, 0.050)
BIN_WALL_H        = 0.055       # REAL: measured with a ruler
BIN_POCKET        = 0.035       # square pocket opening (REAL)
BIN_POCKET_R      = 0.0062      # its corner radius
BIN_FLOOR_T       = 0.030       # REAL: 55 mm tall, pocket 25 mm deep
BIN_WALL_T        = (BIN_OUTER[0] - BIN_POCKET) / 2
BIN_SIZE          = (BIN_POCKET, BIN_POCKET)   # usable interior, for drops
BIN_RELEASE_Z     = SURFACE_DZ + BIN_WALL_H + 0.030  # fingertip at release: 30 mm above the rim

# ------------------------------------------------------------------ cameras
# Cameras are specified as eye + look_at in the parent's frame, not Euler
# angles -- it is the only formulation that stays readable while iterating.

# REAL (2026-10-07): solved from 13 robot poses against two ArUco markers on the
# table (calib_capture.py / calib_solve.py, corner RMS 0.90 px). The lens is
# 163 mm down the tool axis from the J6 origin, 97 mm out on the
# WRISTCAM_DIR side and 12 mm along +Y; the optical axis is within 2 deg of the
# tool axis. Image up is J6 +Z (the side the camera hangs on) -- the old
# (0, 1, 0) rendered the view rolled 90 deg against the real one. fx 620 px
# (datasheet said 597); real k1 -0.020, rendered as a pinhole.
WRIST_CAM = dict(
    parent         = ROBOT_PRIM + "/J6_link",
    eye            = (0.1633, 0.0125, 0.0971),
    look_at        = (0.3632, 0.0180, 0.0930),
    up             = (0.0198, 0.0231, 0.9995),   # flange coordinates
    resolution     = (1280, 720),
    focal_length   = 10.1440,                   # fx 619.6 px at horiz_aperture 20.955
    horiz_aperture = 20.955,
)

# Overhead: Orbbec Gemini 2 color, 1920x1080. REAL (2026-10-07): solved against
# three ArUco markers on the table, two of them located by the wrist camera
# (calib_solve.py, corner RMS 0.42 px). fx 1028 px matches the datasheet
# HFOV 86 deg (960 / tan 43 = 1029); the earlier 1249 px fit was wrong and put
# the mat 100-150 px off. Real k1 +0.009, rendered as a pinhole.
# Eye is 0.777 m forward, -0.043 m to the left, 0.802 m above the table; the
# optical axis meets the table at (0.164, -0.021).
TABLE_CAM = dict(
    parent         = "/World",
    eye            = (0.7770, -0.0426, TABLE_TOP_Z + 0.8024),
    look_at        = (0.1636, -0.0210, TABLE_TOP_Z),
    up             = (-0.7943, -0.0398, 0.6062),
    resolution     = (1920, 1080),
    focal_length   = 11.2158,          # 1027.6 px at horiz_aperture 20.955
    horiz_aperture = 20.955,
    fisheye        = False,
)

# Fixed side webcam: Shenzhen AONI Full HD, 1920x1080.
# REAL (2026-10-08, testbed rebuilt, camera moved): solved from 52 robot poses against the ArUco
# on the gripper (calib_solve.py --free-marker-b --side-k1 0, data/cam_calib_v2, corner RMS
# 3.99 px), as a pinhole since that is how it is rendered (with k1 free: +0.23, RMS 3.33 px).
# 2026-10-07 solve: eye (0.4558, 0.6218, +0.1426), fx 1168.5, RMS 3.08 px.
SIDE_CAM = dict(
    parent         = "/World",
    eye            = (0.4577, 0.6287, TABLE_TOP_Z + 0.1378),
    look_at        = (0.4350, 0.1293, TABLE_TOP_Z + 0.1409),
    up             = (0.0293, 0.0050, 0.9996),
    resolution     = (1920, 1080),
    focal_length   = 12.9427,          # 1185.9 px at horiz_aperture 20.955
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
