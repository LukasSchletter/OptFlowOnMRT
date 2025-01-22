import sys
import subprocess
import collections

# Initialize a deque to store the last 10 lines
last_lines = collections.deque(maxlen=10)

def capture_output(command, stop_marker):
    """Capture output from a terminal command, storing the last 10 lines between start_marker and stop_marker."""
    try:
        # Run the command as a subprocess
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        capturing = True
        while True:
            # Read each line of output
            line = process.stdout.readline()
            
            # If the line is empty, it means the process has finished
            if not line and process.poll() is not None:
                break

            """# Check if the start_marker is encountered and begin capturing
            if start_marker in line:
                capturing = 
                print(f"Started capturing at: {line.strip()}")"""

            # If capturing is enabled, store the line in the deque
            if capturing:
                last_lines.append(line.strip())  # Remove newline characters
                
                # Print the last 10 lines so far
                print("\n".join(last_lines))
                
            # Check if the stop_marker is encountered and stop capturing
            if stop_marker in line:
                print(f"Stopped capturing at: {line.strip()}")
                break

    except Exception as e:
        print(f"Error: {e}")

# Open a file in write mode
log_file = open("output.log", "w")

# Save the current stdout (which is the terminal)
original_stdout = sys.stdout

# Redirect stdout to the file
sys.stdout = log_file

# Print statements will be saved to 'output.log' file
print("This will be saved in the output.log file.")
print("Hello, this is terminal output redirection!")

# Reset stdout to original (the terminal)
sys.stdout = original_stdout

stop_marker = "stop_marker_string" 

command = print("stop_marker_string" )
capture_output(command, stop_marker)
# Now this will print to the terminal
print("This will be printed in the terminal.")




user_input = input("Please enter something: ")

# Print what the user entered
print("You entered:", user_input)

# Close the log file
log_file.close()








    