import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# Step 1: Load 3D Keypoints
def load_keypoints(csv_file):
    """
    Load 3D keypoints from a CSV file into a NumPy array.
    Each row represents a joint, with X, Y, Z coordinates.
    """
    df = pd.read_csv(csv_file)
    keypoints = df[['X', 'Y', 'Z']].values  # Extract X, Y, Z columns

    # Extract X, Y, Z coordinates
    x = df['X']
    y = df['Y']
    z = df['Z']

    # Define the skeleton connections as pairs of point indices
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Left arm
        (1, 5), (5, 6), (6, 7),          # Right arm
        (1, 8),                          # Spine
        (8, 9), (9, 10), (10, 11),       # Left leg
        (8, 12), (12, 13), (13, 14),     # Right leg
        (15, 17), (16, 18),              # Eyes
        (11, 22), (14, 19)               # Feet
    ]

    # Create 3D scatter plot for points
    fig = go.Figure()

    # Add body points
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            text=df.index,  # Use row number as labels
            marker=dict(size=5, color='blue', opacity=0.8),
            name='Body Points'
        )
    )

    # Add skeleton connections
    for start, end in connections:
        fig.add_trace(
            go.Scatter3d(
                x=[x[start], x[end]],
                y=[y[start], y[end]],
                z=[z[start], z[end]],
                mode='lines',
                line=dict(color='red', width=3),
                name='Skeleton'
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # Maintain scale
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25)  # Set initial view position
            )
        ),
        title="3D Skeleton Check"
    )

    # Show the plot
    fig.show()
    return keypoints

# Step 2: Calculate Distance Between Two Points
def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Step 3: Calculate Angle Between Three Points
def calculate_angle(point1, point2, point3):
    """
    Calculate the angle (in degrees) formed by three 3D points.
    point2 is the vertex of the angle.
    """
    vec1 = np.array(point1) - np.array(point2)
    vec2 = np.array(point3) - np.array(point2)
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Ensure cosine value is valid
    print(np.degrees(angle))
    return np.degrees(angle)

# Step 4: Posture Detection Functions
def detect_posture(keypoints):
    """
    Detect and classify posture based on 3D keypoints.
    """
    head = keypoints[0]
    neck = keypoints[1]
    left_shoulder = keypoints[2]
    right_shoulder = keypoints[5]
    left_elbow = keypoints[3]
    right_elbow = keypoints[6]
    left_hand = keypoints[4]
    right_hand = keypoints[7]
    left_hip = keypoints[9]
    right_hip = keypoints[12]
    left_knee = keypoints[10]
    right_knee = keypoints[13]
    left_ankle = keypoints[11]
    right_ankle = keypoints[14]
    spine = keypoints[8]

    # Standing (Neutral)
    if is_standing_neutral(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, head):
        print('Standing')
        if is_jumping_jack(left_shoulder, right_shoulder, left_hand, right_hand, left_ankle, right_ankle, head):
            return "Jumping Jack"
        return "Standing (Neutral)"

    # Sitting
    elif is_sitting(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, head):
        print('Sitting')
        # if is_squat(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
        #     return "Squat"
        return "Sitting"

    # Lying Down
    elif is_lying_down(head, neck, spine, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
        print('Lying Down')
        if is_plank(neck, spine, left_shoulder, right_shoulder, left_elbow, right_elbow, left_hand, right_hand):
            return "Elbow Plank"
        return "Lying Down"

    # Lunges
    elif is_lunge(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
        print('Lunge')
        return "Lunge"

    # Push-Up Variations
    elif is_pushup(neck, spine, left_shoulder, right_shoulder, left_elbow, right_elbow, left_hand, right_hand, left_hip, right_hip, left_knee, right_knee):
        return "Push-Up"

    # Pull-Ups/Chin-Ups
    elif is_pullup(left_hand, right_hand, left_elbow, right_elbow, left_shoulder, right_shoulder, neck, head, spine):
        return "Pull-Up/Chin-Up"

    # Downward Dog
    elif is_downward_dog(left_ankle, right_ankle, left_hip, right_hip, left_hand, right_hand, left_shoulder, right_shoulder, left_knee, right_knee):
        print('Downward Dog')
        return "Downward Dog"

    # Cobra Stretch
    elif is_cobra_stretch(left_ankle, right_ankle, left_hip, right_hip, left_hand, right_hand, left_shoulder, right_shoulder, left_knee, right_knee):
        print('Cobra Stretch')
        return "Cobra Stretch"

    # Jumping
    elif is_jumping(neck, spine, left_ankle, right_ankle):
        return "Jumping"

    # One-Leg Hopping
    elif is_one_leg_hopping(left_ankle, left_knee, right_ankle, right_knee, left_hip, right_hip, head):
        return "One-Leg Hopping"

    else:
        return "Unknown Posture"

# Individual Posture Rules
def is_standing_neutral(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, head):
    """
    Standing Neutral posture: Head is above hips, and hips are above knees.
    """
    if left_hip[1] > right_hip[1]:
        hip1 = right_hip[1]
        hip2 = left_hip[1]
    else:
        hip1 = left_hip[1]
        hip2 = right_hip[1]

    if left_knee[1] > right_knee[1]:
        knee1 = right_knee[1]
        knee2 = left_knee[1]
    else:
        knee1 = left_knee[1]
        knee2 = right_knee[1]
    
    if left_ankle[1] > right_ankle[1]:
        ankle1 = right_ankle[1]
        ankle2 = left_ankle[1]
    else:
        ankle1 = left_ankle[1]
        ankle2 = right_ankle[1]

    # Calculate the average positions of key joints
    hip_avg = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
    knee_avg = [(left_knee[i] + right_knee[i]) / 2 for i in range(3)]
    ankle_avg = [(left_ankle[i] + right_ankle[i]) / 2 for i in range(3)]

    # Calculate angles
    print('Knee Angle: ')
    hip_knee_ankle_angle = calculate_angle(hip_avg, knee_avg, ankle_avg)  # Knee angle
    print('Torso Angle: ')
    head_hip_knee_angle = calculate_angle(head, hip_avg, knee_avg)     

    # Define angle thresholds for a sitting posture
    hip_knee_ankle_range = (160, 200)  # Knee bent ~180°
    head_hip_knee_range = (160, 200)   # Upright torso ~180°

    stand_ang = False
    if (hip_knee_ankle_range[0] <= hip_knee_ankle_angle <= hip_knee_ankle_range[1] and
        head_hip_knee_range[0] <= head_hip_knee_angle <= head_hip_knee_range[1]):
        stand_ang = True
        print('Valid Stand Angles')
    else:
        print('Invalid Stand Angles')

    lst = [head[1], hip1, hip2, knee1, knee2, ankle1, ankle2]
    upright = False
    if np.std([head[1], left_hip[1], right_hip[1], left_ankle[1], right_ankle[1]]) > 50:
        upright = True
    return lst == sorted(lst) and stand_ang and upright

#Todo
def is_jumping_jack(left_shoulder, right_shoulder, left_hand, right_hand, left_ankle, right_ankle, head):
    """
    Jumping Jack: Wide stance and raised arms.
    """
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    ankle_width = calculate_distance(left_ankle, right_ankle)
    return ankle_width > 2 * shoulder_width

def is_sitting(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, head):
    """
    Determines if the posture is sitting based on joint angles.
    Conditions:
    - Hip-Knee-Ankle angle should be between ~80° and ~100° (indicating a bent knee).
    - Head-Hip-Knee angle should be between ~75° and ~105° (indicating an upright torso).
    """
    # Calculate the average positions of key joints
    hip_avg = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
    knee_avg = [(left_knee[i] + right_knee[i]) / 2 for i in range(3)]
    ankle_avg = [(left_ankle[i] + right_ankle[i]) / 2 for i in range(3)]

    if hip_avg[1] > knee_avg[1]:
        lst = [head[1], knee_avg[1], hip_avg[1], ankle_avg[1]]
    else:
        lst = [head[1], hip_avg[1], knee_avg[1], ankle_avg[1]]

    # Calculate angles
    print('Knee Angle: ')
    hip_knee_ankle_angle = calculate_angle(hip_avg, knee_avg, ankle_avg)  # Knee angle
    print('Torso Angle: ')
    head_hip_knee_angle = calculate_angle(head, hip_avg, knee_avg)       # Torso angle

    # Define angle thresholds for a sitting posture
    hip_knee_ankle_range = (80, 100)  # Knee bent ~90°
    head_hip_knee_range = (75, 105)   # Upright torso ~90°

    # Check if both angles are within the defined ranges
    sit_ang = False
    if (hip_knee_ankle_range[0] <= hip_knee_ankle_angle <= hip_knee_ankle_range[1] and
        head_hip_knee_range[0] <= head_hip_knee_angle <= head_hip_knee_range[1]):
        sit_ang = True
        print('Valid sit angle')
    else:
        print('Invalid sit angle')
    return lst == sorted(lst) and sit_ang

def is_lying_down(head, neck, spine, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
    """
    Lying Down: Hips, neck, and ankles are aligned horizontally.
    """
    # Calculate the average positions of key joints
    hip_avg = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
    knee_avg = [(left_knee[i] + right_knee[i]) / 2 for i in range(3)]
    ankle_avg = [(left_ankle[i] + right_ankle[i]) / 2 for i in range(3)]

    # Calculate angles
    print('Knee Angle: ')
    hip_knee_ankle_angle = calculate_angle(hip_avg, knee_avg, ankle_avg)  # Knee angle
    print('Torso Angle: ')
    head_hip_knee_angle = calculate_angle(head, hip_avg, knee_avg)     

    # Define angle thresholds for a sitting posture
    hip_knee_ankle_range = (160, 200)  # Knee bent ~180°
    head_hip_knee_range = (160, 200)   # Upright torso ~180°

    lying_ang = False
    if (hip_knee_ankle_range[0] <= hip_knee_ankle_angle <= hip_knee_ankle_range[1] and
        head_hip_knee_range[0] <= head_hip_knee_angle <= head_hip_knee_range[1]):
        lying_ang = True
        print('Valid Lying Down Angles')
    else:
        print('Invalid Lying Down Angles')
    return np.std([neck[1], spine[1], left_hip[1], right_hip[1], left_ankle[1], right_ankle[1]]) < 50 and lying_ang

# To do
def is_squat(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
    """
    Squat: Hips are lower than knees.
    """
    hip_avg = (left_hip[1] + right_hip[1]) / 2
    knee_avg = (left_knee[1] + right_knee[1]) / 2
    return hip_avg > knee_avg

def is_lunge(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
    """
    Detects lunges (forward or reverse):
    - One leg is bent at a ~90-degree angle at the knee.
    - The other leg is stretched backward or forward.
    """
    left_knee_angle = calculate_angle(left_ankle, left_knee, left_hip)
    right_knee_angle = calculate_angle(right_ankle, right_knee, right_hip)

    # Lunge condition: One knee ~90 degrees, the other leg stretched.
    return (80 <= left_knee_angle <= 100 and right_knee_angle > 120) or \
           (80 <= right_knee_angle <= 100 and left_knee_angle > 120)

def is_plank(neck, spine, left_shoulder, right_shoulder, left_elbow, right_elbow, left_hand, right_hand):
    """
    Detects a plank posture:
    - The body (neck to ankle) is in a straight horizontal line.
    - Angles around shoulder and elbow around 90 deg.
    - Small vertical variations are allowed.
    """
    # Calculate the average positions of key joints
    shld_avg = [(left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)]
    elbow_avg = [(left_elbow[i] + right_elbow[i]) / 2 for i in range(3)]
    hand_avg = [(left_hand[i] + right_hand[i]) / 2 for i in range(3)]

    # Calculate angles
    print('Elbow Angle: ')
    shld_elbow_hand_angle = calculate_angle(shld_avg, elbow_avg, hand_avg) 
    print('Shoulder Angle: ')
    spine_neck_elbow_angle = calculate_angle(spine, neck, elbow_avg)     

    # Define angle thresholds for a sitting posture
    shld_elbow_hand_range = (75, 105)  # Knee bent ~90°
    spine_neck_elbow_range = (75, 105)   # Upright torso ~90°

    plank_ang = False
    if (shld_elbow_hand_range[0] <= shld_elbow_hand_angle <= shld_elbow_hand_range[1] and
        spine_neck_elbow_range[0] <= spine_neck_elbow_angle <= spine_neck_elbow_range[1]):
        plank_ang = True
        print('Valid Plank Angles')
    else:
        print('Invalid Plank Angles')
    return (shld_avg < elbow_avg) and (shld_avg < hand_avg) and plank_ang  # Ensure small vertical variation

# To do
def is_pushup(neck, spine, left_shoulder, right_shoulder, left_elbow, right_elbow, left_hand, right_hand, left_hip, right_hip, left_knee, right_knee):
    """
    Detects push-up variations:
    - The hips are aligned with the shoulders and ankles in a straight line.
    - The knees remain extended.
    """
    hip_to_knee_dist = calculate_distance(left_hip, left_knee) + calculate_distance(right_hip, right_knee)
    return hip_to_knee_dist < 0.1  # Knees extended and aligned

# To do
def is_pullup(left_hand, right_hand, left_elbow, right_elbow, left_shoulder, right_shoulder, neck, head, spine):
    """
    Detects pull-ups or chin-ups:
    - Hands are above the neck, close together.
    """
    wrist_dist = calculate_distance(left_hand, right_hand)
    return wrist_dist < 0.5 and left_hand[1] < neck[1] and right_hand[1] < neck[1]

def is_downward_dog(left_ankle, right_ankle, left_hip, right_hip, left_hand, right_hand, left_shoulder, right_shoulder, left_knee, right_knee):
    """
    Detects the downward dog pose:
    - Hips are significantly raised above the head and ankles.
    """
    # Calculate the average positions of key joints
    hip_avg = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
    knee_avg = [(left_knee[i] + right_knee[i]) / 2 for i in range(3)]
    ankle_avg = [(left_ankle[i] + right_ankle[i]) / 2 for i in range(3)]
    shld_avg = [(left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)]
    hand_avg = [(left_hand[i] + right_hand[i]) / 2 for i in range(3)]

    # Calculate angles
    print('Downward Dog Angle: ')
    dd_angle = calculate_angle(hand_avg, hip_avg, ankle_avg)
    ang1 = calculate_angle(hand_avg, shld_avg, hip_avg)
    ang2 = calculate_angle(hip_avg, knee_avg, ankle_avg)     

    # Define angle thresholds for a sitting posture
    dd_range = (70, 85)  # Knee bent ~75°
    st_range = (160, 200)

    dod_ang = False
    if (dd_range[0] <= dd_angle <= dd_range[1] and
        st_range[0] <= ang1 <= st_range[1] and
        st_range[0] <= ang2 <= st_range[1]):
        dod_ang = True
        print('Valid Downward Dog Angles')
    else:
        print('Invalid Downward Dog Angles')
    return (hip_avg < ankle_avg) and (hip_avg < hand_avg) and (hip_avg < shld_avg) and (hip_avg < knee_avg) and dod_ang

def is_cobra_stretch(left_ankle, right_ankle, left_hip, right_hip, left_hand, right_hand, left_shoulder, right_shoulder, left_knee, right_knee):
    """
    Detects the cobra stretch:
    - The neck and head are significantly raised while hips and knees are grounded.
    """
    # Calculate the average positions of key joints
    hip_avg = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
    knee_avg = [(left_knee[i] + right_knee[i]) / 2 for i in range(3)]
    ankle_avg = [(left_ankle[i] + right_ankle[i]) / 2 for i in range(3)]
    shld_avg = [(left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)]
    hand_avg = [(left_hand[i] + right_hand[i]) / 2 for i in range(3)]

    # Calculate angles
    print('Cobra Stretch Angle: ')
    cs_angle1 = calculate_angle(shld_avg, hip_avg, ankle_avg)
    cs_angle2 = calculate_angle(hand_avg, shld_avg, hip_avg)
    cs_angle3 = calculate_angle(hip_avg, knee_avg, ankle_avg) 

    # Define angle thresholds for a sitting posture
    range1 = (120, 170)
    range2 = (60, 80)
    st_range = (160, 200)

    cs_ang = False
    if (range1[0] <= cs_angle1 <= range1[1] and
        range2[0] <= cs_angle2 <= range2[1] and
        st_range[0] <= cs_angle3 <= st_range[1]):
        cs_ang = True
        print('Valid Cobra Stretch Angles')
    else:
        print('Invalid Cobra Stretch Angles')
    return (shld_avg < ankle_avg) and (shld_avg < hand_avg) and (shld_avg < hip_avg) and (shld_avg < knee_avg) and cs_ang

# To do
def is_jumping(neck, spine, left_ankle, right_ankle):
    """
    Detects a jumping action:
    - Both ankles are significantly above their typical standing height.
    """
    avg_ankle_height = (left_ankle[1] + right_ankle[1]) / 2
    return neck[1] > avg_ankle_height + 0.2  # Ensure ankles are raised

# To do
def is_one_leg_hopping(left_ankle, left_knee, right_ankle, right_knee, left_hip, right_hip, head):
    """
    Detects one-leg hopping:
    - One ankle is grounded, the other ankle is raised above normal standing height.
    """
    grounded_leg = min(left_ankle[1], right_ankle[1])
    raised_leg = max(left_ankle[1], right_ankle[1])
    return raised_leg > grounded_leg + 0.2  # One leg is raised

def posture_feedback(posture, keypoints):
    """
    Provides feedback for correcting posture based on keypoints for each exercise or pose.
    """
    feedback = {}
    
    # Standing Neutral
    if posture == "Standing Neutral":
        shoulder_angle = calculate_angle(keypoints[2], keypoints[1], keypoints[5])  # Shoulders alignment
        hip_angle = calculate_angle(keypoints[8], keypoints[1], keypoints[11])  # Hips alignment
        if abs(shoulder_angle - 180) > 10:
            feedback['shoulders'] = "Keep your shoulders straight and level."
        if abs(hip_angle - 180) > 10:
            feedback['hips'] = "Align your hips with your shoulders."
    
    # Jumping Jack Pose
    elif posture == "Jumping Jack":
        arm_angle = calculate_angle(keypoints[2], keypoints[1], keypoints[4])  # Arms above the head
        if arm_angle < 150:
            feedback['arms'] = "Raise your arms higher above your head."

    # Sitting
    elif posture == "Sitting":
        hip_angle = calculate_angle(keypoints[8], keypoints[1], keypoints[11])
        if hip_angle < 80:
            feedback['hips'] = "Keep your back straight while sitting."

    # Lying Down
    elif posture == "Lying Down":
        hip_height = keypoints[8][1]
        if abs(hip_height - keypoints[11][1]) > 0.1:
            feedback['hips'] = "Keep your hips level with the ground."

    # Squats
    elif posture == "Squat":
        knee_angle = calculate_angle(keypoints[11], keypoints[12], keypoints[13])
        if knee_angle > 90:
            feedback['knees'] = "Lower your hips to get a 90-degree angle at your knees."

    # Jump Squats
    elif posture == "Jump Squat":
        hip_height = keypoints[8][1]
        ankle_height = keypoints[13][1]
        if hip_height < ankle_height + 0.2:
            feedback['jump'] = "Jump higher by extending your legs fully."

    # Lunges
    elif posture == "Lunge":
        left_knee_angle = calculate_angle(keypoints[13], keypoints[12], keypoints[11])
        right_knee_angle = calculate_angle(keypoints[14], keypoints[13], keypoints[8])
        if not (80 <= left_knee_angle <= 100 or 80 <= right_knee_angle <= 100):
            feedback['knees'] = "Ensure your front knee is at a 90-degree angle."

    # Plank
    elif posture == "Plank":
        plank_line = [keypoints[1][1], keypoints[8][1], keypoints[11][1], keypoints[13][1]]
        if np.std(plank_line) > 0.1:
            feedback['alignment'] = "Keep your body in a straight line from head to toe."

    # Push-Up Variations
    elif posture == "Push-Up":
        hip_angle = calculate_angle(keypoints[8], keypoints[1], keypoints[13])
        if hip_angle > 180:
            feedback['hips'] = "Lower your hips to align with your shoulders and feet."
        elif hip_angle < 170:
            feedback['hips'] = "Raise your hips slightly to align with your body."

    # Pull-Ups/Chin-Ups
    elif posture == "Pull-Up/Chin-Up":
        wrist_height = (keypoints[4][1] + keypoints[7][1]) / 2
        neck_height = keypoints[1][1]
        if wrist_height > neck_height:
            feedback['arms'] = "Pull your body higher so that your chin clears the bar."

    # Downward Dog
    elif posture == "Downward Dog":
        hip_height = keypoints[8][1]
        head_height = keypoints[0][1]
        if hip_height < head_height:
            feedback['hips'] = "Raise your hips higher than your head."

    # Cobra Stretch
    elif posture == "Cobra Stretch":
        neck_height = keypoints[1][1]
        hip_height = keypoints[8][1]
        if neck_height > hip_height:
            feedback['neck'] = "Lift your neck and chest higher off the ground."

    # Jumping
    elif posture == "Jumping":
        ankle_height = (keypoints[13][1] + keypoints[14][1]) / 2
        neck_height = keypoints[1][1]
        if neck_height <= ankle_height:
            feedback['jump'] = "Jump higher by fully extending your legs."

    # One-Leg Hopping
    elif posture == "One-Leg Hopping":
        left_ankle_height = keypoints[13][1]
        right_ankle_height = keypoints[14][1]
        if abs(left_ankle_height - right_ankle_height) < 0.2:
            feedback['legs'] = "Raise one leg higher while hopping on the other."

    # Sit-Ups/Leg Raises
    elif posture == "Sit-Ups/Leg Raises":
        knee_angle = calculate_angle(keypoints[11], keypoints[12], keypoints[13])
        if knee_angle < 70 or knee_angle > 110:
            feedback['legs'] = "Keep your legs at a 90-degree angle when raised."
    
    return feedback

def process_directory(directory):
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")
    
    # Get a list of all CSV files in the directory
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    
    if not csv_files:
        raise ValueError("No CSV files found in the specified directory.")
    
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        keypoints = load_keypoints(csv_file)
        print(f"Keypoints for {csv_file}:\n{keypoints}\n")

        posture = detect_posture(keypoints)
        print(f"Detected Posture: {posture}")

        feedback = posture_feedback(posture, keypoints)
        print(f"Feedback: {feedback}")

# Main Function
if __name__ == "__main__":
    process_directory('TestDir')
