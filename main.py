import os

# Run with: nohup python3 main.py &

if __name__=='__main__':
    # Change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Execute Python script
    os.system("python src/models/cnn/supervised_train_script.py")
    # os.system("python src/models/combined_model/supervised_contrastive_train_script.py")
    # os.system("python src/models/mocaplab_fc/supervised_train_script.py")
