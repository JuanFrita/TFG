from abc import ABC, abstractmethod
import os
import shutil
import matplotlib.pyplot as plt
import re
import subprocess

class BaseCNNPipeline(ABC):
    
    ###############################################
    # STRUCTURE SETUP                             #
    ###############################################
    
    def setupDirectories(self, image_source, anotations_source, train_files, val_files, test_files, destination):
        """
        Setups the folders and files for training and testing a cnn

        :image_source: Folder that contains images
        :anotations_source: Folder that contains anotations files
        :train_files: list of train files
        :val_files: list of validation files
        :test_files: list of test files
        :destination: Destination directory of the structure
        """ 
        
        if os.path.exists(destination):
            shutil.rmtree(destination)

        self.createDirectory('train', image_source, anotations_source, train_files, destination)
        self.createDirectory('test', image_source, anotations_source, test_files, destination)
        if val_files is not None:
            self.createDirectory('val',  image_source, anotations_source, val_files, destination)
        
    
    def createDirectory(self, name, image_source, anotations_source, files, destination):
        """
        Build a single directory for a model pipeline

        :name: Name of the directory
        :image_source: Folder that contains images
        :anotations_source: Folder that contains anotations files
        :files: list with files
        :destination: Destination directory of the structure
        """ 
        directory_folder = os.path.join(destination, name)
        
        if not os.path.exists(directory_folder):
            os.makedirs(directory_folder)

        self.loadAnotations(files, directory_folder, image_source, anotations_source)
    
    @abstractmethod
    def loadAnotations(self, files, directory_folder, image_source, anotations_source):
        """
        Load pts files and images into directory_folder

        :files: Files to load
        :directory_folder: Folder destination
        :image_source: Folder that contains images
        :anotations_source: Folder that contains anotations files
        """ 
        pass
    
    def setupListFiles(self, source, destination, extension):
        """
        Creates files with the lists of files for all the model pipeline.
        This is used by models to know the files to use in each pipeline.

        :source: model base path
        :destination: destination file
        :extension: file extension
        """ 
        self.createListFile("train", source, destination, extension)
        self.createListFile("val", source, destination, extension)
        self.createListFile("test", source, destination, extension)

    
    def createListFile(self, name, source, destination, extension):
        """
        Creates a file with the lists of files for a specific pipeline

        :name: pipeline name
        :source: model base path
        :destination: destination file
        :extension: file extension
        """ 
        anotation_folder =  os.path.join(source, name)
        if os.path.exists(anotation_folder) and os.path.isdir(anotation_folder):
            list_file = os.path.join(destination, f"{name}.{extension}")
            self.writeListFile(anotation_folder, list_file)
    
    @abstractmethod
    def writeListFile(self, pipeline_folder, list_file):
        """
        Writes the files in a pipeline into a file list
        
        :pipeline_folder: path to the anotations folder of a model pipeline
        :list_file: file path of list file
        """ 
        pass
    
    ###############################################
    # METRIC VISUALIZATION                        #
    ###############################################
    
    def plotTrainVsValLoss(self, model_name, loss_file, limit_left, limit_right):
        """
        Plot the training and validation loss. 
        
        :model_name: The name of the model
        :loss_file: Path of the file with the loss logs
        :limit_left: Beginning epoch
        :limit_right: Ending epoch
        """
        [train_epochs, train_loss] = self.plot_individual_loss(loss_file, r"train: loss/loss@(\d+): ([\d.]+)", limit_left, limit_right,  True)
        [val_epochs, val_loss] = self.plot_individual_loss(loss_file, r"val: loss/loss@(\d+): ([\d.]+)", limit_left, limit_right)
        plt.plot(train_epochs, train_loss, marker='o', linestyle='-', color='blue', label='Training', markersize=3)
        plt.plot(val_epochs, val_loss, marker='o', linestyle='-', color='red', label='Validation', markersize=3)
        plt.title(f'Training Vs Validation {model_name}')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_individual_loss(loss_file, pattern, limit_left, limit_right, jump=False):
        """
        Plot the data based on a regex expression
        
        :loss_file: Path of the file with the loss logs
        :pattern: Regex expression
        :limit_left: Beginning epoch
        :limit_right: Ending epoch
        :jump: if True jumps based on an interval
        """
        with open(loss_file, 'r') as archivo:
            datos = archivo.read()
        matches = re.findall(pattern, datos)
        if(jump):
            epochs = [int(match[0]) for match in matches[limit_left:limit_right]if int(match[0]) % 5 == 0]
            losses = [float(match[1]) for match in matches[limit_left:limit_right]if int(match[0]) % 5 == 0]
        else:
            epochs = [int(match[0]) for match in matches[limit_left:limit_right//5]]
            losses = [float(match[1]) for match in matches[limit_left:limit_right//5]]
        return [epochs, losses]
    
    ###############################################
    # RUNNING TRAINING AND TESTING                #
    ###############################################
    
    @abstractmethod
    def runTrain(self, data_origin):
        """
        Runs the model training script from its source
        
        :data_origin: Path of training and validation data
        """ 
        pass

    @abstractmethod
    def runTest(self, data_origin, output_dir):
        """
        Runs the model testing script from its source
        
        :data_origin: Path of testing source data
        :output_dir: Path to save testing result
        """
        pass

    ###############################################
    # AUXULIAR METHODS                            #
    ###############################################

    def run_command(self, command):
        """
        Runs a command
        """
        proceso = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, error = proceso.communicate()
        if error:
            print("Error:", error.decode())
        else:
            print("Out:", out.decode())
        return out, error
    
    def getPaths(self, origin):
        """
        Get the absolute paths of a directory
        """
        paths = []
        for file in os.listdir(origin):
            # Usa os.path.join para obtener la ruta absoluta
            paths.append(os.path.abspath(os.path.join(origin, file)))
        return paths
    
    def resetDirectory(path):
        """
        Recreates a directory
        """
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)