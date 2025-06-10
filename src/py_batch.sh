#!/bin/bash

# Bash script to run a Python script multiple times with different arguments
# and capture outputs

# Log file for outputs
LOG_FILE="batch_output.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Clear or create the log file
echo "Batch Run started at $TIMESTAMP" > "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "Starting batch run at $TIMESTAMP"
echo "Logging outputs to $LOG_FILE"
echo ""

# List of parameters to pass to main.py
# Add --use_coe to use quantized coefficients
PARAMS=(
    "python Model_Test.py --eval_data test" 
    "python Model_Test.py --eval_data test --sp_noise --sp_noise_prob 0.05" 
    "python Model_Test.py --eval_data test --sp_noise --sp_noise_prob 0.10" 
    "python Model_Test.py --eval_data test --sp_noise --sp_noise_prob 0.20" 
    "python Model_Test.py --eval_data test --camera_movement" 
    "python Model_Test.py --eval_data test --camera_movement --sp_noise --sp_noise_prob 0.05" 
    "python Model_Test.py --eval_data test --camera_movement --sp_noise --sp_noise_prob 0.10" 
    "python Model_Test.py --eval_data test --camera_movement --sp_noise --sp_noise_prob 0.20" 
)

for PARAM in "${PARAMS[@]}"; do
    echo "Processing parameter: $PARAM"
    echo "Processing parameter: $PARAM" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Run the Python script 10 times for each parameter
    for RUN in {1..10}; do
        echo "Running main.py with parameter: $PARAM (Run $RUN of 10)"
        
        # Run the Python script, capturing stdout and stderr
        STDOUT_FILE=$(mktemp)
        STDERR_FILE=$(mktemp)
        
        START_TIME=$(date +%s.%N)  # Record start time
        $PARAM > "$STDOUT_FILE" 2> "$STDERR_FILE"
        EXIT_CODE=$?  # Capture exit code
        END_TIME=$(date +%s.%N)  # Record end time
        
        # Calculate runtime
        RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)
        
        # Read stdout and stderr
        STDOUT=$(cat "$STDOUT_FILE")
        STDERR=$(cat "$STDERR_FILE")
        
        # Prepare log message
        LOG_MESSAGE="Parameter: $PARAM\nRun: $RUN of 10\nExit Code: $EXIT_CODE\nRuntime: $RUNTIME seconds\nStdout:\n$STDOUT\nStderr:\n$STDERR\n----------------------------------------\n"
        
        # Print to console
        echo -e "$LOG_MESSAGE"
        
        # Append to log file
        echo -e "$LOG_MESSAGE" >> "$LOG_FILE"
        
        # Clean up temp files
        rm "$STDOUT_FILE" "$STDERR_FILE"
    done
    
    # Add a separator between parameters in the log file
    echo "========================================\n" >> "$LOG_FILE"
done

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "Batch run completed at $TIMESTAMP"
echo "Batch run completed at $TIMESTAMP" >> "$LOG_FILE"
