from metaflow import FlowSpec, step, Parameter, conda_base
import os

# @conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2'}, python='3.9.16')
class CropTrainFlow(FlowSpec):
    dataset = Parameter('dataset_name', required = True)
    batch_size = Parameter('batch_size', default = 80)
    learning_rate = Parameter('learning_rate', default = 3e-5)

    @step
    def start(self):
        # check if imageas and cct file existing or not
        print(self.dataset)
        data_pwd = 'images/' + self.dataset
        cct_pwd = f'classifier-training/mdcache/v5.0b/{self.dataset}_cct.json'
        if os.path.exists(data_pwd):
            print(f"Found Data {self.dataset} successfully")
        else:
            print(f"Failed to find Data {self.dataset}")
        if os.path.isfile(cct_pwd):
            print("Found COCO file successfully")
        else:
            print("Failed to find COCO file")
        self.next(self.create_md)

    @step
    def create_md(self):
        os.system(f"python animl-ml/classification/utils/cct_to_md.py \
                    --input_filename classifier-training/mdcache/v5.0b/{self.dataset}_cct.json \
                    --output_filename classifier-training/mdcache/v5.0b/{self.dataset}_md.json\
        ")
        if os.path.isfile(f'classifier-training/mdcache/v5.0b/{self.dataset}_md.json'):
            print("Create a classification label specification JSON file successfully")
        else:
            print("Create a classification label specification JSON file")
        self.next(self.crop)

    @step
    def crop(self):
        os.system(f"python animl-ml/classification/utils/crop_detections.py \
                    classifier-training/mdcache/v5.0b/{self.dataset}_md.json \
                    crops/{self.dataset} \
                    --images-dir ~/images/{self.dataset} \
                    --threshold 0 \
                    --square-crops \
                    --threads 50 \
                    --logdir $BASE_LOGDIR")
        if os.path.exists(f'crops/{self.dataset}'): # also check crop_detections_log???
            print("Images are cropped")
        else:
            print("Images are not cropped")
        self.next(self.train_prep)

    @step
    def train_prep(self):
        os.system(f"python animl-ml/classification/utils/md_to_queried_images.py \
                    --input_filename classifier-training/mdcache/v5.0b/{self.dataset}_md.json \
                    --dataset {self.dataset} \
                    --output_filename $BASE_LOGDIR/queried_images.json")
        base_logdir = os.environ.get('BASE_LOGDIR')
        print(base_logdir)        
        if os.path.exists(base_logdir + '/queried_images.json'):
            print("Create queried_images json file successfully")
        else:
            print("Failed to reate queried_images json file")
        self.next(self.split)
 
    @step
    def split(self):
        os.system("python CameraTraps/classification/create_classification_dataset.py     $BASE_LOGDIR     --mode csv splits     --queried-images-json $BASE_LOGDIR/queried_images.json     --cropped-images-dir ~/crops/     --detector-output-cache-dir ~/classifier-training/mdcache --detector-version 5.0b     --threshold 0     --min-locs 3     --val-frac 0.2     --test-frac 0.2     --method random")
        print("Data are splited into train validation and test sets")     

        # if os.path.exists(base_logdir + '/queried_images.json'):
        #     print("Create queried_images json file successfully")
        # else:
        #     print("Failed to reate queried_images json file")
        self.next(self.train)


    @step
    def train(self):
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        os.system(f"python CameraTraps/classification/train_classifier.py \
                $BASE_LOGDIR \
                ~/crops \
                --model-name efficientnet-b3 --pretrained \
                --label-weighted \
                --epochs 50 --batch-size {batch_size} --lr {learning_rate} \
                --weight-decay 1e-6 \
                --num-workers 4 \
                --logdir $BASE_LOGDIR --log-extreme-examples 3")
        print("model is trained")     
        self.next(self.end)

    @step
    def end(self):
        print("This is the end")


if __name__=='__main__':
    CropTrainFlow()