#!/bin/bash

# Hunyuan3D Safety Mode Test Script
# Tests all safety modes to ensure they launch without segfaults

echo "🧪 Testing Hunyuan3D Safety Modes"
echo "=================================="

# Function to test a specific mode
test_mode() {
    local mode="$1"
    local description="$2"
    
    echo ""
    echo "🔍 Testing $description..."
    echo "Command: ./launch_hunyuan3d.sh $mode"
    
    # Start the process in background and get PID
    timeout 60 ./launch_hunyuan3d.sh $mode &
    local pid=$!
    
    # Wait a bit for startup
    sleep 30
    
    # Check if process is still running
    if kill -0 $pid 2>/dev/null; then
        echo "✅ $description - Server started successfully"
        # Kill the process
        kill $pid 2>/dev/null
        wait $pid 2>/dev/null
        echo "   Server stopped cleanly"
        return 0
    else
        wait $pid 2>/dev/null
        local exit_code=$?
        if [ $exit_code -eq 139 ]; then
            echo "❌ $description - Segmentation fault detected"
        else
            echo "⚠️  $description - Exited with code $exit_code"
        fi
        return $exit_code
    fi
}

# Test results tracking
declare -A results

echo "Starting safety mode tests..."
echo "Each test runs for 30 seconds to verify stable startup"

# Test each mode
test_mode "--minimal" "Minimal mode (no custom rasterizer, no bpy)"
results["minimal"]=$?

test_mode "--no-bpy" "No-BPY mode (Blender functionality disabled)"
results["no-bpy"]=$?

test_mode "--safe" "Safe mode (minimal GPU usage)"
results["safe"]=$?

test_mode "--cpu-only" "CPU-only mode (no GPU rendering)"
results["cpu-only"]=$?

test_mode "" "Full mode (all features enabled)"
results["full"]=$?

# Results summary
echo ""
echo "🎯 Test Results Summary"
echo "======================"

for mode in minimal no-bpy safe cpu-only full; do
    if [ "${results[$mode]}" -eq 0 ]; then
        echo "✅ $mode mode: PASSED"
    elif [ "${results[$mode]}" -eq 139 ]; then
        echo "❌ $mode mode: FAILED (segfault)"
    else
        echo "⚠️  $mode mode: FAILED (exit code ${results[$mode]})"
    fi
done

# Recommendations
echo ""
echo "💡 Recommendations:"
working_modes=()
for mode in minimal no-bpy safe cpu-only full; do
    if [ "${results[$mode]}" -eq 0 ]; then
        working_modes+=("$mode")
    fi
done

if [ ${#working_modes[@]} -eq 0 ]; then
    echo "❌ No modes are working. Check your installation and GPU drivers."
elif [ ${#working_modes[@]} -eq 5 ]; then
    echo "✅ All modes working! Your system is fully compatible."
else
    echo "⚠️  Some modes working. Recommended safe modes:"
    for mode in "${working_modes[@]}"; do
        if [ "$mode" = "full" ]; then
            echo "   ./launch_hunyuan3d.sh (full mode works!)"
        else
            echo "   ./launch_hunyuan3d.sh --$mode"
        fi
    done
fi

echo ""
echo "🔧 For debugging issues, try:"
echo "   ./launch_hunyuan3d.sh --debug --minimal"