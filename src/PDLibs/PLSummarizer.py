import resource 
import numpy as np
import datetime
import os
import pandas as pd
import copy
import sys
import argparse
from sqlalchemy import create_engine
from collections import OrderedDict
import cv2

import runConfigs.PLconfig_grid as PLconfig_grid

from designgines.PLLocationConversion import BinLocation, ThreeDLocation
from designgines.PLLocationConversion import Bin2Plane, ThreeD2Plane

k1 = 10 #enlarge x dimensions
k2 = 3  #enlarge y dimensions

def find_max_row_count(data):
    max_rows = 0
    for k,v in data.items():
        current_row_length = len(data[k])
        if current_row_length > max_rows:
            max_rows = current_row_length
    return max_rows


def ResizeImages(images):
    # Find maximum width and height among all images
    max_width = max(img.shape[1] for img in images)
    max_height = max(img.shape[0] for img in images)
    
    resizedImages = []
    # Calculate offsets for centering images
    for img in images:
        x_offset = (max_width - img.shape[1]) // 2
        y_offset = (max_height - img.shape[0]) // 2

        # Create blank canvas with maximum dimensions
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        
        # Place image onto canvas
        canvas[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

        # Define border parameters
        top_border = 2
        bottom_border = 2
        left_border = 2
        right_border = 2

        # Add a border around the image
        canvas = cv2.copyMakeBorder(
            canvas, top_border, bottom_border, left_border, right_border,cv2.BORDER_CONSTANT, value=(0, 0, 255)
        )  # Red color border (BGR format)

        #append image to list
        resizedImages.append(canvas)
    return resizedImages


class SummarizePlace(object):
    def __init__(self, env, con, design_summary, network_summary, configs_content):
        self.env = env
        self.con = con
        self.design_summary = design_summary
        self.network_summary = network_summary
        self.configs_content = configs_content

        self.twlList = []
        self.twlList_table = {}
        self.twl_episodic_list = []
        self.actionList = []
        self.actionList_table = {}
        self.modeList_table = {}
        self.state_distribution = []
        self.annealing_schedule = []
        self.annealing_schedule_table = {}
        self.total_rewards = 0
        self.total_rewards_episodic_list = []
        self.total_epochs = 0
        self.flog = None
        self.min_index = None
        self.min_value = None
        self.solution_cost = 0
        self.drawing1 = None
            

    def summarize(self):
        self.dump_min_twl_layout()
        self.analyze_twl()
        if self.twlList_table:
            self.analyze_twl_table()
        self.analyze_twl_episodic()
        self.analyze_trewards_episodic()
        self.analyze_actionList()
        if self.actionList_table:
            self.analyze_action_table()
        if self.modeList_table:
            self.analyze_mode_table()
        if self.annealing_schedule:
            self.analyze_annealing_schedule()
        if self.annealing_schedule_table:            
            self.analyze_annealing_schedule_table()
        self.summarize_results()
        self.dump_state_didstribution_file()
        self.dump_stats()
        self.analyze_network_stats()
        self.dump_stats()
        configs_table_dict = self.parse_configs_content()
        self.create_configs_table(configs_table_dict)


    def create_polygons_from_layout(self, newLayout):
        pts_zero = []
        pts_one = []
        names_zero = []
        names_one = []
        x_max = 0
        x_min = float('inf')
        y_max = 0
        y_min = float('inf')
        for _,v in newLayout.nodes.items():
            name = v.name
            z = v.point_lb.z
            x = int(v.point_lb.x)*k1 - 18*k1
            y = (int(v.point_lb.y)*k2 - 18*k2)
            w = int(v.width)*k1
            h = int(v.height)*k2 
            x_max = max(x_max, x+w)
            #x_min = min(x_min, x)
            y_max = max(y_max, y+h)
            #y_min = min(y_min, y)
            pts = [[x+1, y+1], 
                 [x+1, y+h-1], 
                 [x+w-1, y+h-1], 
                 [x+w-1, y+1]
            ]
            #print(pts)
            if int(z) == 1:
                pts_one.append(np.array(pts))
                names_one.append(name)
            else:
                pts_zero.append(np.array(pts))
                names_zero.append(name)

        width = int(x_max + 1)
        height = int(y_max + 1)

        return pts_zero, pts_one, width, height, names_zero, names_one

    def draw_cv2_cluster_images(self, newLayout, clusterMap):
        bImages = self.draw_cv2_image(newLayout, "2d")
        if not bImages: raise ValueError(f"base image not found for clusters")
        if len(bImages) != 1: raise ValueError(f"Expect only one 2D image, found more {len(bImages)}")
        bImage = bImages[0]
        images = []

        print(f"clusterMap = {clusterMap}")
        
        i = 0
        for cluster, clusterData in clusterMap.items():
            if not clusterData['point_lb']: continue
            x = int(clusterData['point_lb'][0]*k1 - 18*k1)
            y = int(clusterData['point_lb'][1]*k2 - 18*k2)
            w = int(clusterData['width']*k1)
            h = int(clusterData['height']*k2)
            image = bImage[y:y+h, x:x+w]
            images.append(image)

            # Save the image
            imageFile = f"placement_cluster{cluster}.png"

            cv2.imwrite(imageFile, image)

            print(f"cluster {cluster} image is {os.path.abspath(imageFile)}")

            i += 1

        images = ResizeImages(images)
        combined_image = np.concatenate(images, axis=1)
        imageCFile = f"placement_cluster_combined.png" 
        cv2.imwrite(imageCFile, combined_image)
        print(f"clusters combined image is {os.path.abspath(imageCFile)}")

    def draw_cv2_image(self, newLayout, tag=""):
        pts_zero, pts_one, width, height, names_zero, names_one = self.create_polygons_from_layout(newLayout)
        images = []
        for i, pts in enumerate([pts_zero, pts_one]):
            if not pts: continue

            images.append('')
            # Create a black image
            images[i] = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw the polygon on the image
            cv2.fillPoly(images[i], pts=pts, color=(0, 255, 0))

            # Define border parameters
            top_border = 2
            bottom_border = 2
            left_border = 2
            right_border = 2

            # Add a border around the image
            images[i] = cv2.copyMakeBorder(
                images[i], top_border, bottom_border, left_border, right_border,cv2.BORDER_CONSTANT, value=(0, 0, 255)
            )  # Red color border (BGR format)


            # Define the text and parameters
            names = names_zero if i == 0 else names_one
            length = len(names)-1
            for ti in [0, length//2, length]:
                text = names[ti]
                org = pts[ti][1]  # Bottom-left corner of the text string in the image
                font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
                font_scale =0.25  # Font scale (size)
                color = (255, 255, 255)  # Font color (white)
                thickness = 1  # Thickness of the text

                # Add the text to the image
                cv2.putText(images[i], text, org, font, font_scale, color, thickness, cv2.LINE_AA)

            scale_x = 3
            scale_y = 3
            images[i] = cv2.resize(images[i], dsize=None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # Save the image
            imageFile = f"placement_layer{i}.png"
            if tag: imageFile = f"placement_{tag}.png"
            cv2.imwrite(imageFile, images[i])
            print(f"layer {i} image is {os.path.abspath(imageFile)}")

        combined_image = np.concatenate(images, axis=1)
        imageCFile = f"placement_combined.png" 
        cv2.imwrite(imageCFile, combined_image)
        print(f"layers combined image is {os.path.abspath(imageCFile)}")

        return images

    def dump_stats(self):
        inputs_layer_ratio = float( self.network_summary['layer_1_nodes'] / self.network_summary['number_of_inputs'])
        content = "layer_inputs_ratio = {}\n\n".format(inputs_layer_ratio)
        content += "minimum TWL = {}\n\n".format(self.env.min_state.last_twl)
        content += "minimum TWL_z = {}\n\n".format(self.env.min_state.last_twl_z)
        content += "minimum TWL_xy = {}\n\n".format(self.env.min_state.last_twl_xy)
        content += "minimum Area = {}\n\n".format(self.env.min_state.last_area)
        content += "minimum Cost = {}\n\n".format(self.env.min_state.twl)
        fh = open('stats.txt', 'a+')
        for k,v in self.design_summary.items():
            content += "{}={}\n".format(k,v)
        for k,v in self.network_summary.items():
            content += "{}={}\n".format(k,v)
        fh.write(content)
        fh.close()

    def dump_min_twl_layout(self):
        # save placement file
        self.env.min_state.dump_layout(PLconfig_grid.designName, os.getcwd())

    def analyze_twl(self):
        #twl lsit and histogram
        print(self.twlList)
        twl_list_file_name = "twl_list_seqpair_{}".format(PLconfig_grid.start)
        twl_hist_file_name = "twl_hist_seqpair_{}".format(PLconfig_grid.start)
        twlarr = Data('TWL', self.twlList, twl_list_file_name, twl_hist_file_name, self.flog)
        
        #get first idex of min twl
        self.min_value = min(self.twlList)
        self.min_index = self.twlList.index(self.min_value)
        self.flog.write('Iteration number to min value %s is %s' % (self.min_value, self.min_index))
        
        twlarr.draw_graphs()


    def summarize_results(self):
        resourceUsageObj = resource.getrusage(resource.RUSAGE_SELF)
        resourceUsageObj_str = "{}".format(resourceUsageObj)
        peakMem = resourceUsageObj.ru_maxrss/1000000
        runTime = datetime.datetime.now().replace(microsecond=0) - PLconfig_grid.start

        print("\nResources used --> Time = {}; Peak Memory = {:.3f} GB ".format(runTime, peakMem))
        self.flog.write("\nResources used --> Time = {}; Peak Memory = {:.3f} GB ".format(runTime, peakMem))
        print("resource.getrusage = {}".format(resourceUsageObj_str.replace(',', '\n')))
        self.flog.write("\nResources used (resource.getrusage) --> \n{} ".format(resourceUsageObj_str.replace(',', '\n')))
        self.flog.close()

        list_of_metrics = []
        labels = ['metric', 'value']
        list_of_metrics.append([0, success_rate])
        #the resourceusage requires pre-processing
        #list_of_metrics.append(['resourceUsageObj',resourceUsageObj_str])
        list_of_metrics.append([1, peakMem])
        list_of_metrics.append([2, runTime.total_seconds()/60])
        
        df = pd.DataFrame(list_of_metrics, columns=labels)
        df.to_sql('performance_metrics_table', self.con, if_exists='append', chunksize=50000, index=False)

        list_of_metrics_mapping = []
        labels = ['id', 'metric_name']
        list_of_metrics_mapping.append([0,'sucessRate'])
        list_of_metrics_mapping.append([1, 'peakMem'])
        list_of_metrics_mapping.append([2, 'runTimeMinutes'])
        df2 = pd.DataFrame(list_of_metrics_mapping, columns=labels)
        df2.to_sql('performance_metrics_mapping_table', self.con, if_exists='append', chunksize=50000, index=False)

    def find_max_height(self, weights_list):
        shape_list = [ x.shape[1] for x in weights_list ]
        return max(shape_list)

    def log_network_calculations(
            self, iteration_string, inputs,
            weights1, biases1, layer_1_output,
            weights2, biases2, layer_2_output,
            weights3, biases3, layer_3_output,
            weights4, biases4, outputs, flag
       ):
        if not flag: return
        #file_name = PLconfig_grid.log_network_operations + "_" + iteration_string + ".csv"
        #file_handle = open(file_name, 'w')
        episodes, epochs = iteration_string.split('_')
        number_of_inputs = self.network_summary['number_of_inputs']
        number_of_outputs = self.network_summary['number_of_outputs']
        #print("number_of_inputs=%s and number_of_outputs=%s" % (number_of_inputs, number_of_outputs))
        output_layers = [ layer_1_output, layer_2_output, layer_3_output, outputs ]
        input_layers = [ inputs, layer_1_output, layer_2_output, layer_3_output ]
        biases_layers = [ biases1, biases2, biases3, biases4 ]
        weights_layers = [ weights1, weights2, weights3, weights4 ]

        # find max height assuming weights are largest # of neurons
        max_number_weights = self.find_max_height(weights_layers)
        weights_header = ''
        for i in range(max_number_weights):
            weights_header += "wtop{},".format(i)
        content = 'episode,epochs,input,' + weights_header +'biases,output,layer,neuron_index'
        labels = content.split(',')
        content += '\n'
        data = []

        for x in range(len(output_layers)):
            #assume weights are the longest structure in the network
            #print("y range=len(weights_layers[{}])=0-{}\n".format(x, len(weights_layers[x])-1))
            
            for y in range(len(weights_layers[x])):
                #print("z range=len(weights_layers[{}][{}])=0-{}\n".format(x,y,len(weights_layers[x][y])-1))
                #for z in range(len(weights_layers[x][y])):
                #print("inputs=", input_layers[x][y][z], "weights", weights_layers[x][y][z]," biases=",biases_layers[x][y], "-q=",output_layers[x][y][z],"-layer=", x,"neuron_index",z, "\n")
                current_number_weights = len(weights_layers[x][y])
                new_weights = weights_layers[x][y]
                if current_number_weights < max_number_weights:
                    diff = max_number_weights - current_number_weights
                    tmp = np.empty(diff)
                    tmp.fill(-999)
                    new_weights = np.concatenate((weights_layers[x][y], tmp))
                weights_list =["{:.3f}".format(x) if isinstance(x, float) else x for x in new_weights.tolist() ]
                weights_string = "{}".format(weights_list)
                try:
                    v1 = input_layers[x][0][y]
                    input_value = "{:.3f}".format(v1) if isinstance(v1, float) else v1
                except:
                    input_value = -999
                try:
                    output_value = output_layers[x][0][y]
                except:
                    output_value = -999
                try:
                    biase_value = biases_layers[x][y]
                except:
                    biase_value = -999
                content += "{},{},{},{},{},{},{},{}\n".format(episodes,epochs,input_value, weights_string[1:-1], biase_value, output_value, x+1, y)
                #wts = weights_string[1:-1].split(',')
                data.append([episodes, epochs, str(input_value)] +  new_weights.tolist() + [ biase_value, output_value, x+1, y])
                #print(new_weights)
        #print(content)
        df = pd.DataFrame(data, columns=labels)
        df.to_sql('forwardPropagation', self.con, if_exists='append', chunksize=50000, index=False)

        #file_handle.write(content)
        #file_handle.close()

    def log_derivative_calculations(self,
            iteration_string,
            q, q_new, loss_value,
            grads_and_vars, 
            new_wts, flag ):
        if not flag: return

        #write loss caclulations csv file
        content = 'episodes, epochs,q_current, q_new, loss\n'
        labels1 = ['episodes','epochs','q_current', 'q_new', 'loss', 'loss_square']
        #file_name = PLconfig_grid.log_loss_operations + "_" + iteration_string + "_" + "{:.0f}".format(loss_value) + ".csv"
        episodes, epochs = iteration_string.split('_')
        #file_handle = open(file_name, 'w')
        assert q.shape == q_new.shape
        data1 = []
        for i in range(len(q[0])):
            diff = q_new[0][i]-q[0][i]
            content += "{},{},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(episodes, epochs, q[0][i], q_new[0][i], diff, diff*diff)
            data1.append([episodes, epochs, q[0][i], q_new[0][i], diff, diff*diff])
        #print(content)

        df1 = pd.DataFrame(data1, columns=labels1)
        df1.to_sql('lossCalculation', self.con, if_exists='append', chunksize=50000, index=False)

        #file_handle.write(content)
        #file_handle.close()

        #write derivative caclulations csv file
        #file_name = PLconfig_grid.log_derivative_operations + "_" + iteration_string + ".csv"
        #file_handle = open(file_name, 'w')
        content = "episodes, epochs, weight,derivative,new_value,layer,neuron_index\n"
        labels = ['episodes','epochs','weight','derivative','new_value','layer','neuron_index']
        data = []
        for i in range(len(grads_and_vars)):
            grads, vars = grads_and_vars[i]
            assert grads.shape == vars.shape
            if len(grads.shape) == 1:
                number_of_inputs = grads.shape[0]
                #print("number_of_inputs=",number_of_inputs,"\n")
                for j in range(number_of_inputs):
                    content += "{},{},{:.3f},{:.3f},{:.3f},{},{}\n".format(episodes, epochs, vars[j],grads[j],vars[j]+grads[j],int(i/2)+1,-1)
                    data.append([int(episodes), int(epochs), float(vars[j]),float(grads[j]),float(vars[j]+grads[j]),int(i/2)+1,int(-1)])
            elif len(grads.shape) == 2:
                number_of_inputs, number_of_neurons = grads.shape[0], grads.shape[1]
                #print("number_of_inputs=",number_of_inputs,"number_of_neurons", number_of_neurons,"\n")
                for j in range(number_of_inputs):
                    for k in range(number_of_neurons):
                        content += "{},{},{:.3f},{:.3f},{:.3f},{},{}\n".format(episodes, epochs, vars[j][k],grads[j][k],vars[j][k]+grads[j][k],int(i/2)+1,k)
                        data.append([int(episodes), int(epochs), float(vars[j][k]),float(grads[j][k]),float(vars[j][k]+grads[j][k]),int(i/2)+1,int(k)])
            else:
                raise ValueError("found unexpected shape of grads and vars = {}".format(len(grads.shape)))

        df = pd.DataFrame(data, columns=labels)
        df.to_sql('reversePropagation', self.con, if_exists='append', chunksize=50000, index=False)
        #print(content)
        #print("new_wts=", new_wts, "\n")
        #file_handle.write(content)
        #file_handle.close()

    def parse_configs_content(self):
        header_template = string(r'^\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*(\S+)\*\*\*\*\*\*\*\*\*\*\*')
        comment_pattern = string(r'^(#|\s)')
        nonparameter_pattern = string(r'^(if )|(else)|(elif)')
        parameter_pattern = string(r'=')
        lines = self.configs_content.split('\n')
        read_contents = {}
        container = None
        flag3 = False
        track_duplicate = []
        for line in lines:
            sline = string(line)
            flag1 = sline.compare(comment_pattern)[0]
            if flag1: continue
            flag2, list2 = sline.compare(header_template)
            if flag2:
                table_name = list2[0].split('/')[1]
                container = read_contents[table_name[:-3]] = []
                flag3 = True
                continue
            if flag3:
                if sline == nonparameter_pattern:
                    continue
                if sline == parameter_pattern:
                    parameter_list = line.split('=')
                    if len(parameter_list) != 2:
                        continue
                    clean_value = parameter_list[1].replace(" ", "").replace("'","")
                    eval_value = clean_value
                    try:
                        eval_value = str(eval(clean_value))
                    except:
                        pass
                    clean_parameter = parameter_list[0].replace(" ", "")
                    if clean_parameter in track_duplicate: 
                        continue
                    track_duplicate.append(clean_parameter)
                    container.append([clean_parameter, eval_value])
        return read_contents

    def create_configs_table(self, configs_table_dict):
        tables_list =  configs_table_dict.keys()
        for table in tables_list:
            table_parameters_list = []
            for count, item in enumerate(configs_table_dict[table]):
                table_parameters_list.append([item[0],count])
                item[0] = count

            df = pd.DataFrame(configs_table_dict[table], columns=['parameter_code', 'value'])
            df.to_sql(table, self.con, if_exists='append', chunksize=50000, index=False)

            mapping_table_name = 'mapping_'+table
            mapping_labels = ['parameter_name', 'value']
            df_mapping = pd.DataFrame(table_parameters_list, columns=mapping_labels)
            df_mapping.to_sql(mapping_table_name, self.con, if_exists='append', chunksize=50000, index=False)

 
def create_op_weights_graph(inputDir):
    from design_manager.PLdm import design_manager

    dm = design_manager()
    nl1 = dm.layout_controller.origLayout

    env = None
    con = None
    summarizer = SummarizePlace(
        env,
        con,
        dm.design_summary,
        PLconfig_grid.network_summary
    )
    os.chdir(inputDir)
    os.mkdir('ouput_weights_comparison')
    os.chdir('ouput_weights_comparison')

    db_path = os.path.join(inputDir,PLconfig_grid.summary_db)
    db_string = 'sqlite:///' + db_path
    engine = create_engine(db_string)

    print("Start generating graphs for network weights")
    start = datetime.datetime.now().replace(microsecond=0)
    PLconfig_grid.log_analyze_oplayerweights_graphs = True
    summarizer.create_oplayer_weights(engine)
    end1 = datetime.datetime.now().replace(microsecond=0)
    print("Time {} taken to complete output layer weights Graph" .format(end1 - start))

def create_all_graphs(inputDir):
    from design_manager.PLdm import design_manager

    dm = design_manager()
    nl1 = dm.layout_controller.origLayout

    env = None
    con = None
    summarizer = SummarizePlace(
        env,
        con,
        dm.design_summary,
        PLconfig_grid.network_summary
    )
    os.chdir(inputDir)

    summarizer.analyze_network_stats()

def main():
    parser = argparse.ArgumentParser()    
        
    parser.add_argument("-inputDir", action="store", dest="inputDir",
                        help="pointer of run directory",
                        required=True, type=str)

    args = parser.parse_args()

    inputDir = os.path.abspath(args.inputDir)

    create_op_weights_graph(inputDir)
    #create_all_graphs(inputDir)

if __name__ == "__main__":
    main()
