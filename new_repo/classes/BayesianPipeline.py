import BaseCNNPipeline
import shutil
import os
import datetime

class BayesianPipeline(BaseCNNPipeline):
    
    ###############################################
    # STRUCTURE SETUP                             #
    ###############################################
    
    def setupDirectories(self, image_source, anotations_source, train_files, val_files, test_files, destination):
        super().setupDirectories(image_source, anotations_source, train_files, val_files, test_files, destination)
        #will override the origin data for the processed one
        self.preprocessData(destination, destination)
    
    def loadAnotations(self, files, directory_folder, image_source, anotations_source):
        for file in files:
            shutil.copy(os.path.join(image_source, file), directory_folder)
            annotation_file = os.path.splitext(file)[0] + '.pts'  #
            shutil.copy(os.path.join(anotations_source, annotation_file), directory_folder)

    def writeListFile(self, source, dest):
        files = self.getPaths(source)
        list_file = open(dest, 'w+')
        for file in files: 
            if file.endswith(".jpg"):
                list_file.write(f'{file}')
                list_file.write('\n')
        list_file.close()
        
    def preprocessData(self, origin, destination):
        """
        Runs preprocessing script for bayesian model
        
        :origin: Origin of models data
        :destination: Directory for bayesian processed data
        """
        command = f"python ../Bayesian-Crowd-Counting-master/preprocess_dataset.py --origin-dir ../new_repo/{origin} --data-dir ../new_repo/{destination}" 
        return self.run_command(command)
    
    ###############################################
    # RUNNING TRAINING AND TESTING                #
    ###############################################
    
    def runTrain(self, data_origin):
        fecha_hora_actual = datetime.now()
        self.train_model(
            f"../new_repo/assets/data_processed/{data_origin}",
            f"../new_repo/assets/results/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}/output",
            100 #same as default
        )
        
    def runTest(self, data_origin, output_dir):
        self.test_model(
            f"../new_repo/assets/data_processed/{data_origin}",
            f"../new_repo/assets/results/{output_dir}",
        )
    
    def trainModel(self, data_root, output_dir, epochs):
        """
        Launches bayesian crowd counting command to train it
        """
        self.resetDirectory(output_dir)
        command = f"python ../Bayesian-Crowd-Counting-master/train.py --data-dir {data_root} --save-dir {output_dir} --max-epoch {epochs}"
        return self.run_command(command)

    def testModel(self, data_root, output_dir):
        """
        Launches bayesian crowd counting command to testit it
        """
        command = f"python ../Bayesian-Crowd-Counting-master/test.py --data-dir {data_root} --save-dir {output_dir}"
        return self.run_command(command)
        
