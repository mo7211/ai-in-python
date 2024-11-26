import sys
import time
import threading

# Function to display the spinner
def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

def spinner_task():
    spinner = spinning_cursor()
    while keep_spinning:
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        sys.stdout.write('\b')
        time.sleep(0.1)

# Your main task function
def long_running_task():
    global keep_spinning
    # Simulate long running task
    time.sleep(10)  # Replace this with the actual work your function does
    keep_spinning = False

if __name__ == "__main__":
    global keep_spinning
    keep_spinning = True
    spinner_thread = threading.Thread(target=spinner_task)

    # Start the spinner in a separate thread
    spinner_thread.start()

    # Run the main task
    long_running_task()

    # Wait for the spinner to finish
    spinner_thread.join()
    print("Task Complete!")