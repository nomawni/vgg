import os


class Remote:

    def __init__(self):
        self.local_dir = os.getenv('LOCAL_DIR')
        self.remote_dir = os.getenv('REMOTE_DIR')

    def upload_directory(self, sftp) -> tuple:

        for filename in os.listdir(self.local_dir):
            local_path = os.path.join(self.local_dir, filename)
            # remote_path = os.path.join(self.remote_dir, filename)
            # Use forward slashes in remote path
            remote_path = self.remote_dir + '/' + filename.replace('\\', '/')

            if os.path.isfile(local_path):
                sftp.put(local_path, remote_path)
            elif os.path.isdir(local_path):
                try:
                    sftp.mkdir(remote_path)
                except IOError:
                    print('The directory probably does not exist ')
                # Modify
                self.local_dir = local_path
                self.remote_dir = remote_path
                self.upload_directory(sftp)
        return local_path, remote_path

    def docker_content(self) -> str:
        return '''
    # Use an official Python runtime as a parent image
    FROM python:3.7-slim

    # Set the working directory in the container
    WORKDIR /app

    # Add current directory contents to the container at /app
    ADD . /app

    # Install any needed packages specified in requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt

    # Make port 80 available to the world outside this container
    EXPOSE 80

    # Run app.py when the container launches
    CMD ["python", "app.py"]
    '''
