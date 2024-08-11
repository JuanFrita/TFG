import BaseCNNPipeline
import os
import shutil
import numpy as np
import datetime

class P2pPipeline(BaseCNNPipeline):
    
    ###############################################
    # STRUCTURE SETUP                             #
    ###############################################
    
    def loadAnotations(self, files, directory_folder, image_source, anotations_source):
        scene = 0
        for file in files:
            string_scene = 'scene0' + \
                str(scene) if scene < 10 else 'scene' + str(scene)

            if not os.path.exists(os.path.join(directory_folder, string_scene)):
                os.makedirs(os.path.join(directory_folder, string_scene))

            # Loads the image and the transformed pts to txt into the directory
            shutil.copy(os.path.join(image_source, file),
                        os.path.join(directory_folder, string_scene, file))

            # transforms pts to txt
            annotation_file = os.path.splitext(file)[0] + '.pts'

            txt_file = os.path.splitext(file)[0] + '.txt'

            self.ptsToTxt(os.path.join(anotations_source, annotation_file),
                              os.path.join(directory_folder, string_scene, txt_file))

            scene += 1
    
    def ptsTotxt(self, pts_file, txt_file):
        """
        Writes the content of a pts file into a txt file for p2p
        training
        """
        write_file = open(txt_file, "w+")
        points = self.read_pts(pts_file)
        for pair in points:  # write the points into a txt
            write_file.write(f'{str(int(pair[0]))} {str(int(pair[1]))}')
            write_file.write('\n')
        write_file.close()
    
    def read_pts(self, path):
        """
        Get the points as array from a pts file
        """
        with open(path) as f:
            rows = [rows.strip() for rows in f]
        head = rows.index('{') + 1
        tail = rows.index('}')
        raw_points = rows[head:tail]
        coords_set = [point.split() for point in raw_points]
        points = np.array([list([float(point) for point in coords])
                           for coords in coords_set]).astype(np.float32)
        return points
    
    def writeListFile(self, pipeline_folder, list_file):
        scenes = self.getPaths(pipeline_folder)
        list_file = open(list_file, 'w+')
        for path in scenes:
            ficheros = self.getPaths(path)
            img = ficheros[0]
            txt = ficheros[1]
            list_file.write(f'{img} {txt}')
            list_file.write('\n')
        list_file.close()
        
    ###############################################
    # RUNNING TRAINING AND TESTING                #
    ###############################################
    
    def runTrain(self, data_origin):
        fecha_hora_actual = datetime.now()
        self.train_p2p_model(
            f"../new_repo/assets/data_processed/{data_origin}",
            100,
            f"../new_repo/assets/results/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}/output",
            f"../new_repo/assets/results/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}/checkpoints",
            f"../new_repo/assets/results/{data_origin}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}/tensorboards",
            6,
            5
        )

    def runTest(self, data_root, output_dir):
        fecha_hora_actual = datetime.now()
        self.test_p2p_model(
            f"../new_repo/assets/results/{output_dir}/checkpoints/best_mae.pth",
            f"../new_repo/assets/data_processed/{data_root}/test",
            f"../new_repo/assets/results/{output_dir}/tests/{data_root}/{fecha_hora_actual.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    
    def trainModel(self, data_root, epochs, output_dir, checkpoints_dir, tensorboard_dir, batch_size, eval_freq):
        """
        Launches p2pnet command to train it
        """
        # create the output dirs
        self.resetDirectory(output_dir)
        self.resetDirectory(tensorboard_dir)
        self.resetDirectory(checkpoints_dir)

        command = f"python ../CrowdCounting-P2PNet-main/train.py --data_root {data_root} --epochs {epochs} --output_dir {output_dir} --checkpoints_dir {checkpoints_dir} --tensorboard_dir {tensorboard_dir} --batch_size {batch_size} --eval_freq {eval_freq} --gpu_id 0"
        return self.run_command(command)

    def testModel(self, weight_path, data_origin, output_dir):
        """
        Launches p2pnet command to testit it
        """
        self.resetDirectory(output_dir)

        command = f"python ../CrowdCounting-P2PNet-main/run_test_bulk.py --weight_path {weight_path} --data_origin {data_origin} --output_dir {output_dir}"
        return self.run_command(command)
    
