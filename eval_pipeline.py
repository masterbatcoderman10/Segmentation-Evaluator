import json
from miseval import evaluate
class EvalPipeline:
    
    def __init__(self, dataloader, n,model_dict, class_list, color_dict):
        
        """"This is an evaluation pipeline which can be used to determine the evaluation metrics for a group of segmentation models."""

        self.dl = dataloader
        self.img, self.gt = zip(*[(data[0].numpy(), data[1].numpy()) for data in dataloader])
        self.img = np.concatenate(self.img)
        #For the methods defined below, the ground-truths need to be in the following shape : (B, H, W, 1)
        self.gt = np.concatenate(self.gt)
        self.gt = np.transpose(self.gt, (0, 2, 3, 1))
        self.gt = np.argmax(self.gt, axis=3)
        self.gt = np.expand_dims(self.gt, axis=-1)
        
        self.model_dict = model_dict
        self.n = n
        self.class_list = class_list
        self.color_dict = color_dict
        self.prediction_gen()
        
    def prediction_gen(self):



        """Generates predictions for each segmentation model"""
        
        predictions = {}
        
        for m in self.model_dict.keys():
            model = self.model_dict[m]

            #For the methods below, the predictions need to be in the following shape : (B, H, W, 1)
            pred = model.predict(self.dl)
            pred = np.argmax(pred, axis=-1)
            pred = np.expand_dims(pred, axis=-1)
            
            predictions[m] = pred
            
        self.pred = predictions
    
    def exact_match_ratio(self, predictions, targets, threshold=0.5):
        """
        :param predictions: tensor of shape (batch_size, num_classes) with predicted scores
        :param targets: tensor of shape (batch_size, num_classes) with target scores
        :return: exact match ratio
        """
        # binarize predictions using threshold
        binary_predictions = torch.where(predictions > threshold, 1, 0)
        # calculate element-wise equality between binary predictions and targets
        equality = torch.eq(binary_predictions, targets)
        # calculate row-wise sums of element-wise equality
        row_sums = torch.sum(equality, dim=1)
        # calculate exact match ratio
        exact_match_ratio = torch.mean(torch.eq(row_sums, targets.shape[1]).float())
        return exact_match_ratio
    
    def jaccard(self, predictions, targets, threshold=0.5):

        binary_predictions = torch.where(predictions > threshold, 1, 0)
        #Calculate the intersection between predictions and targets, sum in the class dimension
        intersection = torch.sum(binary_predictions * targets, dim=1)
        union = torch.sum((binary_predictions | targets), dim=1)

        jaccard_score = torch.mean(intersection.float() / union.float())
        return jaccard_score
    #This function swaps colors from a class map
    def color_swap(self, img):
    
        for key in self.color_dict.keys():

            c = np.where(img[:, :, [0,1,2]] == [key, key, key])
            img[c[0], c[1], :] = self.color_dict[key]

        return img
    
    def image_overlay(self, img, ann):
        
        """This function overlays the annotation over the image
        This function assumes that the background color is pixel value (0,0,0) in RGB
        img : numpy array in the shape : (height, width, color, alpha)
        ann : numpy array in the shape : (height, width, color, alpha)"""
        
        c = np.where(ann[:,:,[0,1,2]] == [0,0,0])
        c_2 = np.where(ann[:,:,[0,1,2]] != [0,0,0])
        ann[c[0], c[1],-1] = 0
        ann[c_2[0], c_2[1],-1] = 100
        
        img = Image.fromarray(img)
        ann = Image.fromarray(ann)
        
        #Overlaying the annotation on the image
        img.paste(ann, (0,0), ann)
        img = np.array(img)
        
        return img
            
    
    def stage_one(self, metrics=["SENS", "SPEC", "IoU", "DSC"], model_keys=[], path="stage_1.csv"):
        
        scores = {}
        
        #First evaluate the same metric for all sets of predictions
        print(metrics)
        for metric in metrics:
            
            scores[metric] = []
            for p in model_keys:
                
                
                current_pred = self.pred[p]
                #Evaluate for the current prediction
                score = evaluate(self.gt, current_pred, metric=metric, multi_class=True, n_classes=self.n)
                score = np.mean(score)
                scores[metric].append(score)
                
        with open(path, "w") as f:
            for m in metrics:
                f.write(m)
                f.write(",")
            f.write("\n") 
            for i in range(len(scores[m])):
                for m in metrics:
                
                    f.write(str(scores[m][i]))
                    f.write(",")
                f.write("\n")
            
            
        return scores
    
    def stage_two(self, metrics=["SENS", "SPEC", "IoU", "DSC"], model_keys=[], path="stage_2.json", csv_path="stage_2.csv"):
        
        scores = {}
        
        # Have to change all of done_pred instances into self.pred
        for p in model_keys:
            scores[p] = []
            print(f"Working on : {p}...")
            for metric in metrics:
                
                current_pred = self.pred[p]
                #Evaluate for the current prediction
                score = evaluate(self.gt, current_pred, metric=metric, multi_class=True, n_classes=self.n)
                scores[p].append(score)
            
            scores[p] = np.array(scores[p]).T
        
        scores_2 = {}
        
        print("Creating final dict")
        with open(csv_path, "w") as csv_file:
            
            #Writing the csv headers
            csv_file.write("model_name")
            csv_file.write(",")
            csv_file.write("class")
            csv_file.write(",")
            
            for metric in metrics:
                csv_file.write(metric)
                csv_file.write(",")
            
            csv_file.write("\n")
            
            for p in model_keys:
                scores_2[p] = {}
                
                for i, c in enumerate(self.class_list):
                    scores_2[p][c] = {}
                    
                    #Starting off the row
                    csv_file.write(p)
                    csv_file.write(",")
                    csv_file.write(c)
                    csv_file.write(",")
                    
                    for n, m in enumerate(metrics):

                        #p : model type
                        #c : class
                        #m : metric
                        
                        #Writing in each metric
                        csv_file.write(str(scores[p][i][n]))
                        csv_file.write(",")
                        
                        scores_2[p][c][m] = scores[p][i][n]
                    
                    csv_file.write("\n")
                        
                    
        
        

        with open(path, "w") as json_file:
            json.dump(scores_2, json_file)
        
        
        return scores_2
    
    def stage_three(self, img_dir, gt_dir, img_files, gt_files, model_keys=[],path="stage_3.png"):
        
        assert len(img_files) == len(gt_files)
        n_cols = len(img_files)
        n_rows = len(model_keys) + 1
        
        #Defining the figure
        fig, ax = plt.subplots(n_cols, n_rows, figsize=(n_rows*3,n_cols*3))
        
        #Plotting the ground truth overlayed on the image
        for i, (img, ann) in enumerate(zip(img_files, gt_files)):
            
            
            
            img_path = os.path.join(img_dir, img)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            
            gt_path = os.path.join(gt_dir, ann)
            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGBA)
            
            overlay_img = self.image_overlay(image, gt)
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2RGB)
            ax[i, 0].imshow(overlay_img)
            ax[i, 0].set_xlabel("ground truth")
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])
        
        for i, model_name in enumerate(model_keys):
            
            i = i+1
            model = self.model_dict[model_name]
            
            for n, img in enumerate(img_files):
                
                img_path = os.path.join(img_dir, img)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (512, 512))
                image = image / 255.0
                image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()
                
                pred = model.predict_image(image)
                pred = np.argmax(pred, axis=-1)
                pred = np.transpose(pred, (1,2,0))
                pred = np.repeat(pred, repeats=3,axis=-1)
                pred = self.color_swap(pred)
                pred = np.uint8(pred)
                pred = cv2.cvtColor(pred, cv2.COLOR_RGB2RGBA)

                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA) 
                image = cv2.resize(image, (512, 512))
                
                overlayed_prediction = self.image_overlay(image, pred)
                overlayed_prediction = cv2.cvtColor(overlayed_prediction, cv2.COLOR_RGBA2RGB)
                
                ax[n, i].imshow(overlayed_prediction)
                ax[n, i].set_xticks([])
                ax[n, i].set_yticks([])
                ax[n, i].set_xlabel(model_name)
        
        plt.subplots_adjust(left=0.2,
                    right=0.9,
                    wspace=0.4,
                    hspace=0.4)
            
        plt.savefig(path, dpi=100)
                    
    
    def stage_four(self, img_dir, gt_dir, img_files, gt_files, model_keys=[],path="stage_4.png"):
        
        """This method is used to compare the predictions of the models passed in with the ground truth for all the images passed in as arguments into this function.
        img_dir : this is the directory of the images : string
        gt_dir : this is the directory where the ground truth images are present : string
        img_files : the specific image file names choosen to plot : list of strings
        gt_files : the corresponding ground truth files choosen to comapare against : list of strings
        The img_files and gt_files must be present within the img_dir and gt_dir respectively.
        Furthermore, the same number of image files and ground truth files should be passed in.
        """
        
        #Number of images should be the same as G.T
        assert len(img_files) == len(gt_files)
        print(img_dir)
        print(gt_dir)
        print(img_files)
        print(gt_files)
        
        n_cols = len(img_files)
        n_rows = len(model_keys) +2
        
        #Defining the figure
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3,n_rows*3))
        
        ax[0, 0].set_ylabel("images")
        ax[1, 0].set_ylabel("ground_truth")
        
        #Plotting the images first
        for i, img in enumerate(img_files):
            
            img_path = os.path.join(img_dir, img)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax[0, i].imshow(image)
            
            ax[0,i].set_xticks([])
            ax[0, i].set_yticks([])
        
        #Plotting the ground truth
        for i, ann in enumerate(gt_files):
            
            gt_path = os.path.join(gt_dir, ann)
            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            #gt = cv2.resize(gt, (512, 512))

            ax[1, i].imshow(gt)
            
            ax[1,i].set_xticks([])
            ax[1, i].set_yticks([])
        
        for i, model_name in enumerate(model_keys):
            
            i = i+2
            model = self.model_dict[model_name]
            ax[i, 0].set_ylabel(model_name)
            
            for n, img in enumerate(img_files):
                
                img_path = os.path.join(img_dir, img)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (512, 512))
                image = image / 255.0
                image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()
                
                pred = model.predict_image(image)
                pred = np.argmax(pred, axis=-1)
                pred = np.transpose(pred, (1,2,0))
                pred = np.repeat(pred, repeats=3,axis=-1)
                pred = self.color_swap(pred)
                
                ax[i, n].imshow(pred)
                ax[i,n].set_xticks([])
                ax[i,n].set_yticks([])
            
        plt.subplots_adjust(left=0.2,
                    right=0.9,
                    wspace=0.4,
                    hspace=0.4)
            
        plt.savefig(path, dpi=100)
    ### Stage 5 : F1 score plots per model per class
    
    def stage_five(self, model_keys=[],path="stage_5.png", csv_path="stage_5.csv"):
        
        f1_scores = []
        
        markers = ["x", "+", ".", "1", "*", "d"]
        colors = ["lime", "fuchsia", "darkorange", "gold", "salmon", "indigo"]
        
        model_names = model_keys
        
        fig, ax = plt.subplots(1,1, figsize=(6,self.n))
        ys = range(self.n)
        
        for i, model_name in enumerate(model_names):
            
            pred = self.pred[model_name]
            
            prec = evaluate(self.gt, pred, metric="PREC", multi_class=True, n_classes=self.n)
            recall = evaluate(self.gt, pred, metric="Recall", multi_class=True, n_classes=self.n)
            
            f1 = (2 * prec * recall) / (prec + recall)
            f1_scores[model_name] = f1
            
            some_num = np.random.uniform(0.1, 0.3, 1)
            
            ax.scatter(x=f1, y=ys, color=colors[i], marker=markers[i], label=model_name)
            ax.set_yticks(ticks=list(range(self.n)),labels=self.class_list)
            ax.legend(loc="best")
            ax.grid()
        
        with open(csv_path, "w") as f:
            
            f.write("Model,")
            for class_name in self.class_list:
                f.write(class_name)
                f.write(",")
                
            f.write("\n")
            
            for i, model_name in enumerate(model_names):
                
                f.write(model_name)
                f.write(",")
                
                f1_score = f1_scores[model_name]
                
                for f1 in f1_score:
                    f.write(f1)
                    f.write(",")
                    
                f.write("\n")
                
            
        plt.savefig(path, dpi=100)

    
    
    def stage_six(self, model_keys=[], path="./stage_6.csv"):
        scores = {}
        for i, model_name in enumerate(model_keys):
            
            pred = torch.array(self.pred[model_name])

            #Getting the global maxpool of the predictions
            clf_pred = f.max_pool2d(pred, pred.shape[2:])
            clf_pred = torch.squeeze(torch.squeeze(clf_pred, dim=-1), dim=-1)

            #Getting the global maxpool of the ground truth
            clf_target_batch = f.max_pool2d(self.gt, self.gt.shape[2:])
            clf_target_batch = torch.squeeze(torch.squeeze(clf_target_batch, dim=-1), dim=-1).float()

            js = self.jaccard(clf_pred, clf_target_batch)
            emr_score = self.exact_match_ratio(clf_pred, clf_target_batch) 

            scores[model_name] = [js, emr_score]
        
        with open(path, "r") as s6:

            s6.write("Model")
            s6.write(",")
            s6.write("Jaccard")
            s6.write(",")
            s6.write("Exact Match Ratio")
            s6.write(",")
            s6.write("\n")

            for i, model_name in enumerate(model_keys):

                s6.write(model_name)
                s6.write(",")

                #Write the jaccard score and exact match ratio
                for metric in scores[model_name]:
                    s6.write(metric)
                    s6.write(",")

                s6.write("\n")

        
        
        
    