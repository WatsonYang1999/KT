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




def calculate_tag(submission):
    # Replace this with your actual logic to calculate the tag
    if submission['judge_score'] == 100:
        return 'Perfect'
    elif submission['judge_result'] == 'AC':
        return 'Accepted'
    else:
        return 'Needs Improvement'


@bp.route('/table/data')
def get_table_data():
    current_app.logger.info("Enter Method get_table_data")
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'Missing userId parameter'}), 400

    current_app.logger.info(user_id)
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT submission_id, problem_id, judge_score, judge_result, submit_time, submit_code 
            FROM submissions
            WHERE student_id = %s
        """

        cursor.execute(query, (user_id,))
        table_data = cursor.fetchall()
        current_app.logger.info(table_data)
        for submission in table_data:
            submission['source_code'] = submission['submit_code']
            submission['tag'] = calculate_tag(submission)

        return jsonify(table_data)

    except Error as e:
        current_app.logger.error(f"Error: {e}")
        return jsonify({'error': 'Database query failed'}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


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

def calculate_performance(problem_id):
    # Add logic to calculate the performance coefficient
    # For now, we'll just return a mock value based on problem_id
    # Replace this with your actual performance calculation logic
    return problem_id % 10 + 1  # Dummy performance coefficient calculation

@bp.route('/problem/performance', methods=['GET'])
def get_problem_performance():

    problem_id = request.args.get('problemId')
    current_app.logger.info(problem_id)
    if not problem_id:
        return jsonify({'error': 'Missing problemId parameter'}), 400

    try:
        problem_id = int(problem_id)
        performance = calculate_performance(problem_id)
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

@bp.route('/performance_prediction/<int:student_id>/<int:problem_id>', methods=['GET'])
def performance_prediction(student_id, problem_id):
    if student_id in student_mastery and problem_id in student_mastery[student_id]:
        mastery = student_mastery[student_id][problem_id]
    else:
        mastery = 0.0  # Default to 0 if no data available

    return jsonify({"student_id": student_id, "problem_id": problem_id, "mastery": mastery}), 200

@bp.route('/image/getImage')
def get_image():
    current_app.logger.info(f"Enter Method getImage")
    image_type = request.args.get('type')
    range_value = request.args.get('range', 10, type=int)

    try:
        range_value = int(range_value)
    except ValueError:
        return jsonify({"error": "Invalid range value"}), 400

    x = np.linspace(-range_value, range_value, 400)

    if image_type == 'A':
        y = np.sin(x)
    elif image_type == 'B':
        y = np.cos(x)
    else:
        return jsonify({"error": "Invalid image type"}), 400

    plt.figure()
    plt.plot(x, y)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    print(f"Image type: {image_type}, Range: {range_value}")  # Echo the parameters for debugging

    return send_file(img, mimetype='image/png')

#code classification
# return a list of string represent the multiple labels of the code-fragment
def code_classify(source_code):
    pass

@bp.route('/mastery_over_time/<int:student_id>', methods=['GET'])
def mastery_over_time(student_id):
    if student_id not in student_mastery:
        return jsonify({"error": "No data for the specified student"}), 404

    time_series = []
    problems = list(student_mastery[student_id].keys())
    mastery_levels = list(student_mastery[student_id].values())

    # Generate a simple time series plot (dummy data for simplicity)
    plt.figure()
    plt.plot(problems, mastery_levels, marker='o')
    plt.title(f'Student {student_id} Mastery Over Time')
    plt.xlabel('Problems')
    plt.ylabel('Mastery Level')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode plot to base64 to send as JSON
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({"student_id": student_id, "mastery_over_time": image_base64}), 200
