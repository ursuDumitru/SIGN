import multiprocessing
import time

# Function to run in a separate process
def process_function(shared_variable):
    while True:
        time.sleep(2)
        shared_variable.value = not shared_variable.value

if __name__ == "__main__":
    # Create a shared variable using multiprocessing's Value
    shared_variable = multiprocessing.Value('b', False)

    # Create a separate process for the function
    process = multiprocessing.Process(target=process_function, args=(shared_variable,))
    process.start()

    # Main thread
    while True:
        # Access the shared variable in the main thread
        print("Shared Variable:", shared_variable.value)
        time.sleep(1)  # Sleep for 1 second to avoid excessive printing

    # Ensure the process terminates when the main program terminates
    process.join()