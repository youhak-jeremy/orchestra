import argparse
import torch
import torch.nn as nn
import torchvision
from models import create_backbone
import utils
from config import get_eval_dict, get_config_dict
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans


def main(config_dict, eval_dict):
    device = torch.device(eval_dict["main_device"] if torch.cuda.is_available() else "cpu")
    # Dataloaders
    _, _, testloader = utils.load_data(config_dict, client_id=-1, bsize=eval_dict["batch_size"], linear_eval=True)
    classes = len(testloader.dataset.classes)
    num_cluster = 256 if classes < 64 else 128

    # Model definitions
    net = create_backbone(name=eval_dict["model_class"], num_classes=0).to(device)

    # Load model
    pretrained_model = torch.load(eval_dict["pretrained_loc"], map_location='cpu')
    net.load_state_dict({k[9:]:v for k, v in pretrained_model['net'].items() if k.startswith('backbone.')}, strict=True)    
    del pretrained_model
    net = net.to(device)
    net.eval()

    # Accumulate embeddings
    accumulated_embeddings = []
    raw_test_y = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = net(inputs.to(device))
            accumulated_embeddings.extend([x for x in outputs])
            raw_test_y.extend([x for x in targets])
    tmp = torch.stack(accumulated_embeddings, 0)
    if device != 'cpu':
        tmp = tmp.cpu()
    # cluster_ids_x, cluster_centers = kmeans(
    #             X=tmp, num_clusters=num_cluster, distance='euclidean', device=device
    #         )
    clustering = KMeans(n_clusters=num_cluster, random_state=0).fit(tmp)
    cluster_centers = torch.from_numpy(clustering.cluster_centers_)
    cluster_ids_x = clustering.labels_

    print ("Finished clustering!")

    # Test        
    confusion_mat = torch.zeros((num_cluster,10))
    # pred_clusters = cluster_ids_x.numpy()
    pred_clusters = cluster_ids_x 
    correct = 0
    total = len(accumulated_embeddings)
    for i, pred_cluster in enumerate(pred_clusters):
        confusion_mat[pred_cluster, raw_test_y[i]] += 1
    
    for i, pred_cluster in enumerate(pred_clusters):
        pred_label = torch.argmax(confusion_mat[pred_cluster])
        correct += (pred_label == raw_test_y[i])

    print("Accuracy: {} %".format(correct.numpy()/total*100))

def get_parser():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--do_linear", type=bool, default=True)
    parser.add_argument("--config_dict", type=str, default="{}")
    parser.add_argument("--eval_dict", type=str, default="{}")
    return parser

def update_configs(args, config_dict, eval_dict):
    config_dict_update = eval(args.config_dict)
    config_dict.update(config_dict_update)

    eval_dict_update = eval(args.eval_dict)
    eval_dict.update(config_dict_update)
    eval_dict.update(eval_dict_update)
    return config_dict, eval_dict

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config_dict = get_config_dict()
    eval_dict = get_eval_dict()
    config_dict, eval_dict = update_configs(args, config_dict, eval_dict)
    main(config_dict, eval_dict)
