from flask import Blueprint, request, jsonify,send_file,current_app
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import ast
import mysql.connector
from mysql.connector import Error
bp = Blueprint('routes', __name__)
from server.app.kt_reference import infer,get_chart_input
# In-memory store for submissions
submissions = []

# In-memory store for student mastery (for simplicity)
student_mastery = {}



def get_db_connection():
    host_name = "localhost"
    user_name = "root"
    user_password = "17377756"
    db_name = "buaa_online_judge"
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

def get_student_record(user_id):
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT submission_id, problem_id, judge_score, judge_result, submit_time, submit_code 
            FROM submissions
            WHERE student_id = %s
            ORDER BY problem_id ASC
        """

        cursor.execute(query, (user_id,))
        table_data = cursor.fetchall()
        return table_data
    except Error as e:
        current_app.logger.error(f"Error: {e}")
        return jsonify({'error': 'Database query failed'}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def table_data_to_record(input_table_data):
    print(input_table_data)
    records = []
    for submission in input_table_data:
        records.append((submission['problem_id'],submission['judge_score']))
    return records

def calculate_tag(submission):
    from server.app.source_code_inference  import get_tags
    # Replace this with your actual logic to calculate the tag
    current_app.logger.info("Enter Method Calculate Tags")
    try:
        tags = get_tags(submission['source_code'])

    except Exception as e:
        print('Exception Occurred')
        tags = ['null']

    return tags.__str__()


@bp.route('/table/data')
def get_table_data():
    current_app.logger.info("Enter Method get_table_data")
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'Missing userId parameter'}), 400

    current_app.logger.info(user_id)

    try:
        table_data = get_student_record(user_id)
        current_app.logger.info(table_data)
        for submission in table_data:
            submission['source_code'] = submission['submit_code']
            submission['tag'] = calculate_tag(submission)

        return jsonify(table_data)

    except Error as e:
        current_app.logger.error(f"Error: {e}")
        return jsonify({'error': 'Database query failed'}), 500



@bp.route('/addTableEntry', methods=['POST'])
def add_table_entry():
    data = request.json
    current_app.logger.info("Enter Method add table entry ",data)

    if 'entry' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    try:
        user_id = data['userId']
        new_entry = ast.literal_eval(data['entry'])
        if not isinstance(new_entry,
                          dict) or 'submission_id' not in new_entry or 'problem_id' not in new_entry or 'judge_score' not in new_entry or 'judge_result' not in new_entry or 'submit_time' not in new_entry or 'source_code' not in new_entry:
            raise ValueError
    except (ValueError, SyntaxError):
        return jsonify({
                           'error': 'Invalid entry format. Expected format: {"submission_id": 1, "problem_id": 1, "judge_score": 100, "judge_result": "AC", "submit_time": "YYYY-MM-DD HH:MM:SS", "source_code": "code"}'}), 400

    connection = get_db_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO submissions (submission_id, student_id, problem_id, contest_id, language_id, submit_time, submit_code, judge_score, judge_result)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # add missed field
    new_entry['student_id'] = user_id
    new_entry['contest_id'] = 1
    new_entry['language_id'] = 3
    current_app.logger.info(new_entry)
    cursor.execute(query, (
    new_entry['submission_id'], new_entry['student_id'], new_entry['problem_id'], new_entry['contest_id'],
    new_entry['language_id'], new_entry['submit_time'], new_entry['source_code'], new_entry['judge_score'],
    new_entry['judge_result']))

    connection.commit()

    new_entry['tag'] = calculate_tag(new_entry)

    cursor.close()
    connection.close()

    return jsonify(new_entry), 201

@bp.route('/add_submission', methods=['POST'])
def add_submission():
    data = request.json
    current_app.logger.info("Enter Method add submission ",data)
    required_fields = ['student_id', 'submission_id', 'problem_id', 'source_code', 'judge_result', 'time_stamp']

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing fields in the submission data"}), 400

    submissions.append(data)

    # Update student mastery (dummy update logic for simplicity)
    student_id = data['student_id']
    problem_id = data['problem_id']
    judge_result = data['judge_result']

    if student_id not in student_mastery:
        student_mastery[student_id] = {}

    if judge_result == "Accepted":
        student_mastery[student_id][problem_id] = 1.0
    else:
        student_mastery[student_id][problem_id] = 0.0

    return jsonify({"message": "Submission added successfully"}), 201

# def calculate_performance(problem_id):
#
#     # Add logic to calculate the performance coefficient
#     # For now, we'll just return a mock value based on problem_id
#     # Replace this with your actual performance calculation logic
#     return problem_id % 10 + 1  # Dummy performance coefficient calculation

@bp.route('/problem/performance', methods=['GET'])
def get_problem_performance():
    user_id = request.args.get('userId')
    problem_id = request.args.get('problemId')

    current_app.logger.info(problem_id)
    if not problem_id:
        return jsonify({'error': 'Missing problemId parameter'}), 400

    try:
        problem_id = int(problem_id)
        table_data = get_student_record(user_id)
        current_app.logger.info(table_data)

        records = table_data_to_record(table_data)
        performance = infer(records,problem_id)
        return jsonify({'problem_id': problem_id, 'performance': performance})
    except ValueError:
        return jsonify({'error': 'Invalid problemId parameter'}), 400


@bp.route('/submission/<int:submission_id>/source_code', methods=['GET'])
def get_source_code(submission_id):
    current_app.logger.info(f"Fetching source code for submission_id: {submission_id}")
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT submit_code FROM submissions WHERE submission_id = %s"
        cursor.execute(query, (submission_id,))
        submission = cursor.fetchone()

        if not submission:
            return jsonify({'error': 'Submission not found'}), 404

        return jsonify(submission)

    except Error as e:
        current_app.logger.error(f"Error: {e}")
        return jsonify({'error': 'Database query failed'}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# @bp.route('/performance_prediction/<int:student_id>/<int:problem_id>', methods=['GET'])
# def performance_prediction(student_id, problem_id):
#     if student_id in student_mastery and problem_id in student_mastery[student_id]:
#         mastery = student_mastery[student_id][problem_id]
#     else:
#         mastery = 0.0  # Default to 0 if no data available
#
#     return jsonify({"student_id": student_id, "problem_id": problem_id, "mastery": mastery}), 200

# 示例数据
scores = {
    's1': [1, 2, 3, 4],
    's2': [3, 4, 5, 6],
    's3': [4, 5, 6, 7],
    's4': [2, 3, 4, 5]
}

def line_chart(scores):
    plt.figure(figsize=(10, 5))
    for subject, score in scores.items():
        plt.plot(score, label=subject, marker='o')

    plt.xlabel('Exercise Number')
    plt.ylabel('Scores')
    plt.title('Scores in Each Subject')
    plt.legend()
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    return img

def radar_chart(scores):
    # 转换数据
    subjects = list(scores.keys())
    values = np.array(list(scores.values()))
    num_vars = len(subjects)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = np.concatenate((values, values[:, [0]]), axis=1)
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i in range(values.shape[0]):
        ax.plot(angles, values[i], linewidth=2, linestyle='solid', label=f'skill {i+1}')
        ax.fill(angles, values[i], alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subjects)
    ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7'])

    plt.title('Radar Chart of Scores')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    img = io.BytesIO()
    plt.savefig(img, format='png')
    return img

@bp.route('/image/getImage')
def get_image():
    user_id = request.args.get('userId')
    current_app.logger.info(f"Enter Method getImage")
    image_type = request.args.get('type')
    range_value = request.args.get('range', 10, type=int)
    #todo : modify range value
    try:
        range_value = int(range_value)
    except ValueError:
        return jsonify({"error": "Invalid range value"}), 400

    table_data = get_student_record(user_id)
    current_app.logger.info(table_data)

    records = table_data_to_record(table_data)
    scores = get_chart_input(records,range_value)
    if image_type == 'A':

        img = line_chart(scores)
    elif image_type == 'B':
        img = radar_chart(scores)
    else:
        return jsonify({"error": "Invalid image type"}), 400



    # Save the plot to a BytesIO object

    img.seek(0)

    print(f"Image type: {image_type}, Range: {range_value}")  # Echo the parameters for debugging

    return send_file(img, mimetype='image/png')


