

def calculate_time(start_time, end_time) -> None:

    elapsed_time_seconds = end_time - start_time

    # Convert seconds to minutes and hours
    elapsed_minutes = elapsed_time_seconds // 60
    elapsed_seconds = elapsed_time_seconds % 60

    elapsed_hours = elapsed_minutes // 60
    elapsed_minutes = elapsed_minutes % 60

    print('Total time for training: {} hours, {} minutes, {} seconds'.format(
        int(elapsed_hours), int(elapsed_minutes), int(elapsed_seconds)))
