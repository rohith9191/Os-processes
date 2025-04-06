from flask import Flask, render_template, request
import psutil
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from datetime import datetime

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Constants
PAST_INTERVALS = 25  # Fixed number of intervals for past data

def get_system_uptime():
    return time.time() - psutil.boot_time()

def format_time_values(time_values):
    time_unit = "Seconds"
    if max(time_values) >= 60:
        time_values = time_values / 60
        time_unit = "Minutes"
    if max(time_values) >= 60:
        time_values = time_values / 60
        time_unit = "Hours"
    return time_values, time_unit

def get_past_system_metrics():
    uptime = get_system_uptime()
    time_points = np.linspace(0, uptime, num=PAST_INTERVALS)
    time_points, time_unit = format_time_values(time_points)

    history = []
    for idx, t in enumerate(time_points, start=1):
        cpu_usage = psutil.cpu_percent(interval=0.5)
        memory_usage = psutil.virtual_memory().percent
        time_with_unit = f"{round(t, 2)} {time_unit[:3]}"
        history.append([idx, time_with_unit, cpu_usage, memory_usage])

    df = pd.DataFrame(history, columns=["Index", "Time (Unit)", "CPU Usage", "Memory Usage"])
    df.set_index("Index", inplace=True)
    return df, time_unit

def generate_future_time_series(total_period_min, interval_min):
    intervals = int(total_period_min / interval_min)
    return [f"{i * interval_min} Min" for i in range(1, intervals + 1)]

def predict_future_trends(past_data, total_period_hours=1, interval_min=6):
    total_period_min = total_period_hours * 60
    X = np.array(range(1, len(past_data) + 1)).reshape(-1, 1)
    y_cpu = np.array(past_data["CPU Usage"]).reshape(-1, 1)
    y_mem = np.array(past_data["Memory Usage"]).reshape(-1, 1)

    model_cpu = LinearRegression().fit(X, y_cpu)
    model_mem = LinearRegression().fit(X, y_mem)

    future_intervals = int(total_period_min / interval_min)
    future_indices = np.array(range(1, future_intervals + 1)).reshape(-1, 1)
    
    future_cpu = model_cpu.predict(future_indices).flatten()
    future_mem = model_mem.predict(future_indices).flatten()

    cpu_variation = np.std(y_cpu) * np.random.uniform(-0.5, 0.5, size=len(future_cpu))
    mem_variation = np.std(y_mem) * np.random.uniform(-0.3, 0.3, size=len(future_mem))

    future_cpu = np.clip(future_cpu + cpu_variation, 0, 100)
    future_mem = np.clip(future_mem + mem_variation, 0, 100)

    future_times = generate_future_time_series(total_period_min, interval_min)

    future_data = pd.DataFrame({
        "Index": range(1, future_intervals + 1),
        "Time (Unit)": future_times,
        "Predicted CPU Usage": np.round(future_cpu, 2),
        "Predicted Memory Usage": np.round(future_mem, 2),
    }).set_index("Index")

    return future_data

def detect_bottlenecks(past_data, future_data):
    current_bottlenecks = []
    future_bottlenecks = []
    sustained_bottlenecks = []
    
    # Threshold values
    critical_threshold = 90
    warning_threshold = 80
    notice_threshold = 70
    
    # CURRENT BOTTLENECKS
    current_cpu_avg = past_data["CPU Usage"].mean()
    current_cpu_peak = past_data["CPU Usage"].max()
    current_mem_avg = past_data["Memory Usage"].mean()
    current_mem_peak = past_data["Memory Usage"].max()
    
    if current_cpu_peak >= critical_threshold:
        current_bottlenecks.append(f"🚨 CRITICAL: Current CPU peaked at {current_cpu_peak}%")
    elif current_cpu_peak >= warning_threshold:
        current_bottlenecks.append(f"⚠️ WARNING: Current CPU peaked at {current_cpu_peak}%")
    elif current_cpu_peak >= notice_threshold:
        current_bottlenecks.append(f"ℹ️ NOTICE: Current CPU peaked at {current_cpu_peak}%")
    
    if current_mem_peak >= critical_threshold:
        current_bottlenecks.append(f"🚨 CRITICAL: Current Memory peaked at {current_mem_peak}%")
    elif current_mem_peak >= warning_threshold:
        current_bottlenecks.append(f"⚠️ WARNING: Current Memory peaked at {current_mem_peak}%")
    elif current_mem_peak >= notice_threshold:
        current_bottlenecks.append(f"ℹ️ NOTICE: Current Memory peaked at {current_mem_peak}%")
    
    # FUTURE BOTTLENECKS
    future_cpu_peak = future_data["Predicted CPU Usage"].max()
    future_mem_peak = future_data["Predicted Memory Usage"].max()
    
    if future_cpu_peak >= critical_threshold:
        future_bottlenecks.append(f"🔮 CRITICAL: Future CPU may peak at {future_cpu_peak}%")
    elif future_cpu_peak >= warning_threshold:
        future_bottlenecks.append(f"🔮 WARNING: Future CPU may peak at {future_cpu_peak}%")
    elif future_cpu_peak >= notice_threshold:
        future_bottlenecks.append(f"🔮 NOTICE: Future CPU may peak at {future_cpu_peak}%")
    
    if future_mem_peak >= critical_threshold:
        future_bottlenecks.append(f"🔮 CRITICAL: Future Memory may peak at {future_mem_peak}%")
    elif future_mem_peak >= warning_threshold:
        future_bottlenecks.append(f"🔮 WARNING: Future Memory may peak at {future_mem_peak}%")
    elif future_mem_peak >= notice_threshold:
        future_bottlenecks.append(f"🔮 NOTICE: Future Memory may peak at {future_mem_peak}%")
    
    # SUSTAINED BOTTLENECKS (both present and future)
    if (past_data["CPU Usage"] > warning_threshold).any() and (future_data["Predicted CPU Usage"] > warning_threshold).any():
        sustained_bottlenecks.append("🔥 SUSTAINED: Continuous high CPU usage detected")
    if (past_data["Memory Usage"] > warning_threshold).any() and (future_data["Predicted Memory Usage"] > warning_threshold).any():
        sustained_bottlenecks.append("🔥 SUSTAINED: Continuous high Memory usage detected")
    
    # Format the output
    all_bottlenecks = []
    
    if current_bottlenecks:
        all_bottlenecks.append("=== CURRENT SYSTEM STATUS ===")
        all_bottlenecks.extend(current_bottlenecks)
    
    if future_bottlenecks:
        all_bottlenecks.append("=== FUTURE PREDICTIONS ===")
        all_bottlenecks.extend(future_bottlenecks)
    
    if sustained_bottlenecks:
        all_bottlenecks.append("=== SUSTAINED PATTERNS ===")
        all_bottlenecks.extend(sustained_bottlenecks)
    
    return all_bottlenecks if all_bottlenecks else ["✅ No significant bottlenecks detected"]

def suggest_optimizations(past_data, future_data):
    optimizations = []
    
    # Current CPU optimizations
    current_cpu = past_data["CPU Usage"].mean()
    if current_cpu > 85:
        optimizations.append("🔴 IMMEDIATE ACTION: CPU overload - Close unnecessary processes")
    elif current_cpu > 70:
        optimizations.append("🟠 RECOMMENDATION: High CPU usage - Optimize running applications")
    
    # Current Memory optimizations
    current_mem = past_data["Memory Usage"].mean()
    if current_mem > 85:
        optimizations.append("🔴 IMMEDIATE ACTION: Memory critical - Close applications or add RAM")
    elif current_mem > 70:
        optimizations.append("🟠 RECOMMENDATION: High memory usage - Check for memory leaks")
    
    # Future optimizations
    future_cpu = future_data["Predicted CPU Usage"].mean()
    if future_cpu > 80:
        optimizations.append("🔮 PLANNING: Rising CPU usage - Consider hardware upgrades")
    
    future_mem = future_data["Predicted Memory Usage"].mean()
    if future_mem > 80:
        optimizations.append("🔮 PLANNING: Increasing memory pressure - Plan for RAM upgrade")
    
    # General optimizations
    optimizations.append("💡 TIP: Regular system maintenance improves performance")
    optimizations.append("💡 TIP: Monitor background processes for resource usage")
    
    return optimizations

@app.route('/', methods=['GET', 'POST'])
def index():
    total_period_hours = 1
    interval_min = 6
    
    if request.method == 'POST':
        total_period_hours = float(request.form.get('total_period_hours', 1))
        interval_min = int(request.form.get('interval_min', 6))
    
    past_data, time_unit = get_past_system_metrics()
    future_data = predict_future_trends(past_data, total_period_hours, interval_min)
    bottlenecks = detect_bottlenecks(past_data, future_data)
    optimizations = suggest_optimizations(past_data, future_data)
    
    past_table = past_data.to_html(
        classes='table table-striped table-hover', 
        justify='left',
        float_format='%.2f',
        border=0,
        index_names=False
    )
    
    future_table = future_data.to_html(
        classes='table table-striped table-hover',
        justify='left',
        float_format='%.2f',
        border=0,
        index_names=False
    )
    
    return render_template(
        'index.html',
        past_table=past_table,
        future_table=future_table,
        bottlenecks=bottlenecks,
        optimizations=optimizations,
        total_period_hours=total_period_hours,
        interval_min=interval_min,
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, extra_files=['templates/index.html'])
