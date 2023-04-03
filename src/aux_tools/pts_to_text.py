import numpy as np

file_pts = 'C:/Users/Usuario/TFG/src/resources/pts_list/pts_petrained'
write_prefix = 'C:/Users/Usuario/TFG/resources/Test_Training_p2p/test_pretrained/test'

def read_pts(path):
    with open(path) as f:
        rows = [rows.strip() for rows in f]
    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')
    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]
    """Convert entries from lists of strings to tuples of floats"""
    points = np.array([list([float(point) for point in coords]) for coords in coords_set]).astype(np.float32)
    return points

def write_doc():
    #read pts files
    file1 = open(file_pts, 'r')
    Lines = file1.readlines()
    for line in Lines: #parse them into txt for p2pnet
        line = line.strip()
        write_file = open(f'{write_prefix}/{line.split("/")[len(line.split("/"))-1].replace(".pts", ".txt")}', "w+")
        print(line)
        points = read_pts(line)
        for pair in points: #write the points into a txt
            write_file.write(f'{str(int(pair[0]))} {str(int(pair[1]))}')
            write_file.write('\n')
        write_file.close()

write_doc()