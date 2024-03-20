import os
# import subprocess

from src.models.cnn import rgbd_object_cnn_supervised_training

# Run with: nohup python3 main.py &

def test():
    # Execute Python script
    # https://www.geeksforgeeks.org/run-one-python-script-from-another-in-python/
    # https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
    # https://stackoverflow.com/questions/4965159/how-to-redirect-output-with-subprocess-in-python
    script = "src/models/cnn/supervised_train_script.py"
    cmd = ["python3", script]
    log_file = "supervised_cnn.log"
    print(f"Executing: pyhton3 {script} > {log_file}")
    # os.system(f"python3 {script} > {log_file}")
    # with open(log_file, "a") as out_file:
    #     subprocess.run(cmd, stdout=out_file)

if __name__=='__main__':
    # Change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    rgbd_object_cnn_supervised_training()
