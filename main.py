import os

# Run with: nohup python3 main.py &

if __name__=='__main__':
    # Change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Execute Python script
    script = "src/models/cnn/supervised_train_script.py"
    log = "supervised_cnn.log"
    print(f"Executing: pyhton3 {script} > {log}")
    os.system(f"python3 {script} > {log}")
    # os.system("python3 src/models/combined_model/supervised_contrastive_train_script.py")
    # os.system("python3 src/models/mocaplab_fc/supervised_train_script.py")
