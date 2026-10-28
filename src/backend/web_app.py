"""
Tire Production Monitoring Web Application
Real-time monitoring of tire drum production with YOLO detection
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, time
import os
import json
from pathlib import Path
import secrets

app = Flask(__name__)
# Use a fixed secret key to persist sessions across restarts
# In production, this should be stored in environment variables
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tire_production.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')  # admin or user
    created_at = db.Column(db.DateTime, default=datetime.now)

class Shift(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    is_deleted = db.Column(db.Boolean, default=False)  # Soft delete flag
    deleted_at = db.Column(db.DateTime, nullable=True)  # When was it deleted
    created_at = db.Column(db.DateTime, default=datetime.now)

class TireProduction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    daily_number = db.Column(db.Integer)  # Day-wise tire number (resets daily)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    duration_seconds = db.Column(db.Integer)
    status = db.Column(db.String(20))  # in_process, completed
    confidence_avg = db.Column(db.Float)
    photos = db.Column(db.Text)  # JSON array of photo paths
    created_at = db.Column(db.DateTime, default=datetime.now)
    shift_id = db.Column(db.Integer, db.ForeignKey('shift.id'), nullable=True)
    shift = db.relationship('Shift', backref='tires')

class ProductionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    detection_class = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    frame_path = db.Column(db.String(255))
    tire_id = db.Column(db.Integer, db.ForeignKey('tire_production.id'))

class SystemStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stream_active = db.Column(db.Boolean, default=False)
    last_update = db.Column(db.DateTime, default=datetime.now)
    current_detection = db.Column(db.String(50))
    frames_processed = db.Column(db.Integer, default=0)
    rtsp_url = db.Column(db.String(255))

class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    minimum_duration_minutes = db.Column(db.Integer, default=5)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

# Template Filters
@app.template_filter('time_12h')
def time_12h_filter(dt):
    """Convert datetime to 12-hour format"""
    if dt is None:
        return ''
    return dt.strftime('%I:%M:%S %p')

@app.template_filter('datetime_12h')
def datetime_12h_filter(dt):
    """Convert datetime to 12-hour format with date"""
    if dt is None:
        return ''
    return dt.strftime('%Y-%m-%d %I:%M:%S %p')

@app.template_filter('duration_hhmm')
def duration_hhmm_filter(seconds):
    """Convert seconds to readable format (e.g., 9m 5s or 3h 9m 5s)"""
    if seconds is None:
        return ''
    seconds = int(seconds)  # Convert to int if it's a float
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    else:
        return f'{minutes}m {secs}s'

# Helper function to get next daily tire number
def get_next_daily_number():
    """Get the next daily tire number (resets each day)"""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    
    # Count tires created today
    count = TireProduction.query.filter(
        TireProduction.created_at >= today_start,
        TireProduction.created_at < today_end
    ).count()
    
    return count + 1

# Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get today's statistics
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_tires = TireProduction.query.filter(
        TireProduction.created_at >= today_start,
        TireProduction.status == 'completed'
    ).all()
    
    # Get system status
    system_status = SystemStatus.query.first()
    if not system_status:
        system_status = SystemStatus()
        db.session.add(system_status)
        db.session.commit()
    
    # Calculate statistics
    total_today = len(today_tires)
    total_duration = sum([t.duration_seconds or 0 for t in today_tires])
    avg_duration = total_duration / total_today if total_today > 0 else 0
    
    # Get current status
    current_tire = TireProduction.query.filter_by(status='in_process').first()
    
    # Get current shift (only active shifts)
    current_shift = None
    now_time = datetime.now().time()
    shifts = Shift.query.filter_by(is_deleted=False).all()
    for shift in shifts:
        if shift.start_time <= shift.end_time:
            if shift.start_time <= now_time <= shift.end_time:
                current_shift = shift
                break
        else:
            if now_time >= shift.start_time or now_time <= shift.end_time:
                current_shift = shift
                break
    
    return render_template('dashboard.html',
                         total_today=total_today,
                         avg_duration=avg_duration,
                         current_tire=current_tire,
                         system_status=system_status,
                         current_shift=current_shift,
                         username=session.get('username'))

@app.route('/production')
def production():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get filter parameters
    date_filter = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    start_time_filter = request.args.get('start_time', '')
    end_time_filter = request.args.get('end_time', '')
    shift_filter = request.args.get('shift_id', '')
    
    filter_date = datetime.strptime(date_filter, '%Y-%m-%d')
    
    query = TireProduction.query
    
    # Base date filter
    start_dt = filter_date
    end_dt = filter_date + timedelta(days=1)
    
    # Apply time range within the date if provided
    if start_time_filter:
        try:
            st = datetime.strptime(start_time_filter, '%H:%M').time()
            start_dt = datetime.combine(filter_date, st)
        except ValueError:
            pass
            
    if end_time_filter:
        try:
            et = datetime.strptime(end_time_filter, '%H:%M').time()
            # If end time is before start time, assume it means next day (though purely time filtering on same date is ambiguous)
            # For simplicity, let's assume strict time on the selected date
            end_dt_candidate = datetime.combine(filter_date, et)
            if start_time_filter and end_dt_candidate < start_dt:
                 # If end time < start time, it might mean overnight, but here we are filtering by "Date". 
                 # Let's assume the user wants time range on that specific date.
                 pass
            else:
                end_dt = end_dt_candidate
        except ValueError:
            pass

    query = query.filter(
        TireProduction.created_at >= start_dt,
        TireProduction.created_at < end_dt
    )
    
    if shift_filter and shift_filter.isdigit():
        query = query.filter(TireProduction.shift_id == int(shift_filter))
    
    tires = query.order_by(TireProduction.created_at.desc()).all()
    
    # Only show active shifts in filter dropdown
    shifts = Shift.query.filter_by(is_deleted=False).all()
    
    return render_template('production.html',
                         tires=tires,
                         selected_date=date_filter,
                         selected_start_time=start_time_filter,
                         selected_end_time=end_time_filter,
                         selected_shift=int(shift_filter) if shift_filter.isdigit() else None,
                         shifts=shifts,
                         username=session.get('username'))

@app.route('/tire/<int:tire_id>')
def tire_detail(tire_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    tire = TireProduction.query.get_or_404(tire_id)
    
    # Parse photos JSON
    photos = json.loads(tire.photos) if tire.photos else []
    
    # Get logs for this tire
    logs = ProductionLog.query.filter_by(tire_id=tire_id).order_by(ProductionLog.timestamp).all()
    
    return render_template('tire_detail.html',
                         tire=tire,
                         photos=photos,
                         logs=logs,
                         username=session.get('username'))

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get statistics for last 7 days
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    daily_stats = []
    for i in range(7):
        day = datetime.now() - timedelta(days=6-i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        count = TireProduction.query.filter(
            TireProduction.created_at >= day_start,
            TireProduction.created_at < day_end,
            TireProduction.status == 'completed'
        ).count()
        
        daily_stats.append({
            'date': day.strftime('%Y-%m-%d'),
            'count': count
        })
    
    # Get shift-wise statistics for today
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    
    from sqlalchemy import func
    
    # Include tires with shifts (including deleted ones) - use LEFT JOIN
    shift_stats = db.session.query(
        Shift.name, 
        func.count(TireProduction.id)
    ).join(TireProduction, TireProduction.shift_id == Shift.id)\
     .filter(TireProduction.created_at >= today_start, 
             TireProduction.created_at < today_end,
             TireProduction.status == 'completed')\
     .group_by(Shift.name).all()
    
    # Also count tires with no shift (shift was deleted and not soft-deleted, or created without shift)
    no_shift_count = TireProduction.query.filter(
        TireProduction.created_at >= today_start,
        TireProduction.created_at < today_end,
        TireProduction.status == 'completed',
        TireProduction.shift_id == None
    ).count()
    
    shift_data = [{'name': name, 'count': count} for name, count in shift_stats]
    
    # Add "No Shift" category if there are tires without shifts
    if no_shift_count > 0:
        shift_data.append({'name': 'No Shift', 'count': no_shift_count})
    
    return render_template('analytics.html',
                         daily_stats=daily_stats,
                         shift_data=shift_data,
                         username=session.get('username'))

@app.route('/settings')
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
    
    system_status = SystemStatus.query.first()
    users = User.query.all()
    
    # Get or create system settings
    system_settings = SystemSettings.query.first()
    if not system_settings:
        system_settings = SystemSettings(minimum_duration_minutes=5)
        db.session.add(system_settings)
        db.session.commit()
    
    return render_template('settings.html',
                         system_status=system_status,
                         system_settings=system_settings,
                         users=users,
                         username=session.get('username'))

# API Endpoints
@app.route('/api/status')
def api_status():
    """Get current system status for real-time updates"""
    system_status = SystemStatus.query.first()
    current_tire = TireProduction.query.filter_by(status='in_process').first()
    
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = TireProduction.query.filter(
        TireProduction.created_at >= today_start,
        TireProduction.status == 'completed'
    ).count()
    
    # Always use temp_frame.jpg as the latest frame
    latest_frame = 'temp_frame.jpg'
    
    return jsonify({
        'stream_active': system_status.stream_active if system_status else False,
        'current_detection': system_status.current_detection if system_status else 'unknown',
        'current_tire_id': current_tire.id if current_tire else None,
        'current_tire_duration': (datetime.now() - current_tire.start_time).seconds if current_tire else 0,
        'today_count': today_count,
        'last_update': system_status.last_update.isoformat() if system_status else None,
        'latest_frame': latest_frame
    })

@app.route('/api/update_rtsp', methods=['POST'])
def update_rtsp():
    """Update RTSP stream URL"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    rtsp_url = data.get('rtsp_url')
    
    system_status = SystemStatus.query.first()
    if not system_status:
        system_status = SystemStatus()
        db.session.add(system_status)
    
    system_status.rtsp_url = rtsp_url
    system_status.last_update = datetime.now()
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/update_minimum_duration', methods=['POST'])
def update_minimum_duration():
    """Update minimum production duration threshold"""
    # Check if user is logged in
    if 'user_id' not in session:
        print(f"[ERROR] update_minimum_duration: No user_id in session")
        return jsonify({'error': 'Not logged in'}), 401
    
    # Check if user is admin
    user_role = session.get('role')
    print(f"[INFO] update_minimum_duration: user_id={session.get('user_id')}, username={session.get('username')}, role={user_role}")
    
    if user_role != 'admin':
        print(f"[ERROR] update_minimum_duration: User role '{user_role}' is not admin")
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.json
    minimum_duration = data.get('minimum_duration_minutes')
    
    if not minimum_duration or not isinstance(minimum_duration, int) or minimum_duration < 1 or minimum_duration > 99:
        return jsonify({'error': 'Invalid duration (must be 1-99 minutes)'}), 400
    
    system_settings = SystemSettings.query.first()
    if not system_settings:
        system_settings = SystemSettings()
        db.session.add(system_settings)
    
    system_settings.minimum_duration_minutes = minimum_duration
    system_settings.updated_at = datetime.now()
    db.session.commit()
    
    print(f"[SUCCESS] Updated minimum_duration_minutes to {minimum_duration}")
    return jsonify({'success': True, 'minimum_duration_minutes': minimum_duration})

@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    """Start the RTSP stream processing"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
    
    system_status = SystemStatus.query.first()
    if system_status:
        system_status.stream_active = True
        system_status.last_update = datetime.now()
        db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    """Stop the RTSP stream processing"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
    
    system_status = SystemStatus.query.first()
    if system_status:
        system_status.stream_active = False
        system_status.last_update = datetime.now()
        db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/orphaned_tires', methods=['GET'])
def get_orphaned_tires():
    """Get list of orphaned tires (stuck in_process)"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
    
    if session.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    orphaned = TireProduction.query.filter_by(status='in_process').all()
    
    result = []
    for tire in orphaned:
        duration_minutes = (datetime.now() - tire.start_time).total_seconds() / 60
        detection_count = ProductionLog.query.filter_by(tire_id=tire.id).count()
        
        result.append({
            'id': tire.id,
            'start_time': tire.start_time.isoformat(),
            'duration_minutes': round(duration_minutes, 1),
            'detection_count': detection_count,
            'has_photos': bool(tire.photos)
        })
    
    return jsonify({'orphaned_tires': result})

@app.route('/api/complete_tire/<int:tire_id>', methods=['POST'])
def complete_tire_manual(tire_id):
    """Manually complete a stuck tire"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
    
    if session.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    tire = TireProduction.query.get(tire_id)
    if not tire:
        return jsonify({'error': 'Tire not found'}), 404
    
    if tire.status != 'in_process':
        return jsonify({'error': 'Tire is not in_process'}), 400
    
    # Complete the tire
    tire.end_time = datetime.now()
    tire.duration_seconds = (tire.end_time - tire.start_time).seconds
    tire.status = 'completed'
    
    # Calculate average confidence
    logs = ProductionLog.query.filter_by(tire_id=tire.id).all()
    if logs:
        tire.confidence_avg = sum([log.confidence for log in logs]) / len(logs)
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': f'Tire #{tire.daily_number} completed'})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve captured images"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # If requesting temp_frame.jpg, serve from webapp directory
    if filename == 'temp_frame.jpg':
        base_dir = Path(__file__).parent
        return send_from_directory(base_dir, filename)
    
    # Serve other images from results/predictions directory
    base_dir = Path(__file__).parent.parent / 'results' / 'predictions'
    return send_from_directory(base_dir, filename)

@app.route('/api/shifts', methods=['GET', 'POST'])
def api_shifts():
    """Manage shifts"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    if request.method == 'GET':
        # Only return active (non-deleted) shifts for shift management
        shifts = Shift.query.filter_by(is_deleted=False).order_by(Shift.start_time).all()
        return jsonify([{
            'id': s.id,
            'name': s.name,
            'start_time': s.start_time.strftime('%H:%M'),
            'end_time': s.end_time.strftime('%H:%M')
        } for s in shifts])
    
    if request.method == 'POST':
        data = request.json
        name = data.get('name')
        start_time_str = data.get('start_time')
        end_time_str = data.get('end_time')
        
        if not all([name, start_time_str, end_time_str]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        try:
            start_time = datetime.strptime(start_time_str, '%H:%M').time()
            end_time = datetime.strptime(end_time_str, '%H:%M').time()
        except ValueError:
            return jsonify({'error': 'Invalid time format (HH:MM)'}), 400
            
        shift = Shift(name=name, start_time=start_time, end_time=end_time)
        db.session.add(shift)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 400
            
        return jsonify({'success': True, 'id': shift.id})

@app.route('/api/shifts/<int:shift_id>', methods=['DELETE'])
def delete_shift(shift_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
        
    shift = Shift.query.get_or_404(shift_id)
    
    # Commercial best practice: Check if shift has production data
    tire_count = TireProduction.query.filter_by(shift_id=shift.id).count()
    
    if tire_count > 0:
        # Soft delete: Mark as deleted but preserve data
        shift.is_deleted = True
        shift.deleted_at = datetime.now()
        # Make name unique by appending timestamp to avoid conflicts
        shift.name = f"{shift.name} (Deleted {datetime.now().strftime('%Y-%m-%d %H:%M')})"
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'soft_delete': True,
            'message': f'Shift marked as deleted. {tire_count} tire(s) will show this shift as "(Deleted)"'
        })
    else:
        # Hard delete: No production data, safe to remove completely
        db.session.delete(shift)
        db.session.commit()
        return jsonify({
            'success': True,
            'hard_delete': True,
            'message': 'Shift deleted permanently (no production data)'
        })

# Initialize database and create default admin user
def init_db():
    with app.app_context():
        db.create_all()
        
        # Migration: Check and add columns to TireProduction if missing
        inspector = inspect(db.engine)
        columns = [c['name'] for c in inspector.get_columns('tire_production')]
        
        if 'shift_id' not in columns:
            print("Migrating database: Adding shift_id to tire_production")
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE tire_production ADD COLUMN shift_id INTEGER REFERENCES shift(id)"))
                conn.commit()
        
        if 'daily_number' not in columns:
            print("Migrating database: Adding daily_number to tire_production")
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE tire_production ADD COLUMN daily_number INTEGER"))
                conn.commit()
            print("✓ daily_number column added successfully")
        
        # Migration: Add soft delete columns to Shift table
        shift_columns = [c['name'] for c in inspector.get_columns('shift')]
        
        if 'is_deleted' not in shift_columns:
            print("Migrating database: Adding is_deleted to shift")
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE shift ADD COLUMN is_deleted BOOLEAN DEFAULT 0"))
                conn.commit()
            print("✓ is_deleted column added successfully")
        
        if 'deleted_at' not in shift_columns:
            print("Migrating database: Adding deleted_at to shift")
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE shift ADD COLUMN deleted_at DATETIME"))
                conn.commit()
            print("✓ deleted_at column added successfully")
        
        # Default-admin credential provisioning intentionally excluded from portfolio.\n        \n        # Create default system settings if not exists
        settings = SystemSettings.query.first()
        if not settings:
            settings = SystemSettings(minimum_duration_minutes=5)
            db.session.add(settings)
            db.session.commit()
            print("Default system settings created: minimum_duration = 5 minutes")
