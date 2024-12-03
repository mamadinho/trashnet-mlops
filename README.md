# TrashNet-MLOps

A journey of learning MLOps, as my past professional experience does not utilize MLOps well.
A publicly available [Trash Classification Dataset](https://huggingface.co/datasets/garythung/trashnet) is used as its training dataset.

Available features:
- **Cross Validation**: Able to do cross validation using pre-defined folds, learning rate (LR) and batch size. A plot consists of loss and accuracy is saved to determine whether it is worth to try using the determined hyperparameter. Sample of cross validation results:
![image](https://github.com/user-attachments/assets/55de4975-25a3-48d7-9f65-914f5ca710b1)
- **Full Training**: After finding that a set of parameters works for our deep learning model ([ConvNextV2](https://huggingface.co/timm/convnextv2_atto.fcmae)), we can train using the whole training dataset. WanDB logging is optional to log several metrics, including confusion matrix for both training and validation dataset. A plot of confusion matrix and sample of misclassified for each possible classification is saved. Example of misclassified visualization:
![image](https://github.com/user-attachments/assets/e4ad2dc5-b2c7-4190-acd5-713be93f917f)
- **Plot Misclassified Manually**: Ability to see which data is classified as X when it should be classified as Y.
![image](https://github.com/user-attachments/assets/c79377e0-de13-4d48-9698-f5965b312057)
- **GradCAM**: To see which features is 'seen' by the model to determine the data is classified as some class X. Example of GradCAM on **Metal** data but classified as **Glass**:
![image](https://github.com/user-attachments/assets/f5f4788f-a57e-48b6-97aa-8c073c8be288)

WanDB is used for logging the metrics, and it is optional. If you want to use it, make sure you have logged in on your local WanDB and change the config into the name that you want. 

![image](https://github.com/user-attachments/assets/95d901b9-4a0b-4c39-946e-900c45374123)

The best performing model is available at the model registry at WanDB. Note: This should be automated to select the best performing model, since my time is limited currently I set it manually using the website and choose a model.
![image](https://github.com/user-attachments/assets/feea3ed3-a449-463f-a555-7e8388987445)

The best model is available to access using **HuggingFace**
