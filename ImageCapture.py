import cv2
import torch
from torchvision import transforms
from PIL import Image
from CNN_Resnet import CNNModel

#Load the saved model
hyperpara = {
    'out_features': 29,
    'lr': 1e-3,
}
model = CNNModel(**hyperpara)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load('models/final_resnet.pth', device=device)
video = cv2.VideoCapture(0)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

softmax = torch.nn.Softmax(dim=1)



def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def predict_frame(img_tensor):
    prediction = model.predict_one_step(img_tensor)
    s = softmax(prediction)
    class_id = torch.argmax(s)
    # print(classes[class_id])
    return class_id


def try_image(img_path="Hello11.jpg"):
    Ima = Image.open(img_path)
    im = Ima.resize((200,200))
    img_tensor = transforms.ToTensor()(im).unsqueeze_(0)
    img_tensor = img_tensor.unsqueeze_(0)
    class_id = predict_frame(img_tensor)
    print(classes[class_id])


while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        
        im = crop_center(im, 500, 500)
        im = im.resize((200,200))
        img_tensor = transforms.ToTensor()(im).unsqueeze_(0)
        img_tensor = img_tensor.unsqueeze_(0)
  
        # class_id = predict_frame(img_tensor)
        
        # cv2.putText(img=frame, text=f"{classes[class_id]}", org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
        cv2.imshow("Capturing", frame)

        k  = cv2.waitKey(1)
        # if the escape key is been pressed, the app will stop
        if k%256 == 27:
            print('escape hit, closing the app')
            break
        # if the spacebar key is been pressed
        # screenshots will be taken
        elif k%256  == 32:
            im = Image.fromarray(frame, 'RGB')
            im.save("Hello11.jpg")

            class_id = predict_frame(img_tensor)
            print(classes[class_id])
            print('screenshot taken')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()