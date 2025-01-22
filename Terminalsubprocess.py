import subprocess
import collections

def capture_last_ten_lines(command):
    """Capture and store the last 10 lines of terminal output from a command."""
    # Create a deque with a maximum length of 10 to store the last 10 lines
    last_lines = collections.deque(maxlen=10)

    try:
        # Run the command as a subprocess and capture its output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Read the output line by line
        for line in process.stdout:
            last_lines.append(line.strip())  # Remove newline characters and store the line
            # Print the last 10 lines (or fewer if there are not enough)
            print("\n".join(last_lines))

        # Optionally capture stderr (standard error)
        stderr_output = process.stderr.read().strip()
        if stderr_output:
            print(f"Error: {stderr_output}")

    except Exception as e:
        print(f"Error executing the command: {e}")

def example_function_to_run_in_terminal():
    """Example function that produces terminal output."""
    # This is just an example; replace this with your actual function
    print("Starting process...")
    print("Processing data...")
   
    print("Process complete!")

command = "python -c 'import sys; sys.stdout.write(\"Starting process...\\nProcessing data...\\n\" + \"\\n\".join([f\"Line {i+1}: Some terminal output\" for i in range(20)]) + \"\\nProcess complete!\\n\")'"
if __name__ == "__main__":
    # Capture the output of the example function
    # We'll redirect the output of `example_function_to_run_in_terminal` to a subprocess
    capture_last_ten_lines("python -c 'import sys; sys.stdout.write(\"Starting process...\\nProcessing data...\\n\" + \"\\n\".join([f\"Line {i+1}: Some terminal output\" for i in range(20)]) + \"\\nProcess complete!\\n\")'")
