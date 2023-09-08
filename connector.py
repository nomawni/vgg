from __future__ import barry_as_FLUFL
import paramiko
import os
from remote import Remote
from dotenv import load_dotenv
import subprocess


class Connector(Remote):

    def __init__(self) -> None:
        super().__init__()
        self.ssh = paramiko.SSHClient()
        self.private_key_path = r'C:\Users\kolon\Documents\MI-Project\.ssh\id_ed25519'

    def get_private_key_str(self) -> str:
        with open(self.private_key_path, "r") as f:
            private_key_str = f.read()

        return private_key_str

    def save_requirements(self) -> None:
        with open('app/requirements.txt', 'a') as f:
            subprocess.run(['pip', 'freeze'], stdout=f)

    def connect(self):
        private_key = paramiko.Ed25519Key(
            filename=self.private_key_path, password='KolonMIProject90!*')
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.ssh.connect('141.13.17.218', username='user80', passphrase='KolonMIProject90!*',
                         pkey=private_key)
        stdin, stdout, stderr = self.ssh.exec_command('docker image ls')
        # Wait until the command is finished executing
        exit_status = stdout.channel.recv_exit_status()

        if exit_status == 0:
            print('Printed the docker images successfully !')
        else:
            print("Error", exit_status)
        # Transfer the model file to local machine
        sftp = self.ssh.open_sftp()
        # sftp.get('/home/user80/Dockerfile',
        #         r'C:\Users\kolon\Documents\MI-Project\project\Dockerfile')
        # First save all the modules that have been installed
        self.save_requirements()
        # Upload the app directory to the remote server
        local_path, remote_path = self.upload_directory(sftp=sftp)
        dockerfile_content = self.docker_content()
        sftp_file = sftp.open(f'{remote_path}/Dockerfile', 'w')
        sftp_file.write(dockerfile_content)
        sftp_file.close()

        stdin, stdout, stderr = self.ssh.exec_command(
            f'docker build -t mi-project:1.0 {remote_path}')
        print(stdout.read().decode())
        print(stderr.read().decode())

        stdin, stdout, stderr = self.ssh.exec_command(
            'docker run -p 8081:80 mi-project:1.0')

        print(stdout.read().decode())
        print(stderr.read().decode())
        sftp.close()

        self.ssh.close()


if __name__ == "__main__":
    load_dotenv()
    connector = Connector()
    connector.connect()
