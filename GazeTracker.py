import numpy as np
import cv2
import pygame
import time
import sys

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#  initialize the variables for monitoring the directions

delta_X = 0
delta_Y = 0
right = 0
left = 0
up = 0
down = 0
center = 0
eye_blink = 0
X_position = 50
Y_position = 500
select = False

# Capture video from camera0
cap = cv2.VideoCapture(0)

# Initialize Pygame 
pygame.init()

# Set running as True, to stay in loop
running = True

# Set screen size vector
size = width, height = 1300,800

# Create an 'screen' object, which receives the size vector as parameter
screen = pygame.display.set_mode(size)

# Initializing colors as RGB
black = (0, 0, 0)
white = (255,255,255)
red = (255,0,0)

# Load .wav sound files for Interface

sound1 = pygame.mixer.Sound('cima.wav')
sound2 = pygame.mixer.Sound('baixo.wav')
sound3 = pygame.mixer.Sound('esquerda.wav')
sound4 = pygame.mixer.Sound('direita.wav')
sound5 = pygame.mixer.Sound('bomdia.wav')
sound6 = pygame.mixer.Sound('fome.wav')
sound7 = pygame.mixer.Sound('banheiro.wav')
sound8 = pygame.mixer.Sound('sono.wav')
sound9 = pygame.mixer.Sound('frio.wav')
sound10 = pygame.mixer.Sound('calor.wav')
sound11 = pygame.mixer.Sound('sede.wav')

# Load .png images for GUI

frio = pygame.image.load('frio.png')
fome = pygame.image.load('fome.png')
sono = pygame.image.load('sono.png')
calor = pygame.image.load('calor.jpg')
agua = pygame.image.load('sede.png')
banheiro = pygame.image.load('banheiro.png')

# This Function evaluates a point, returning a value for its probability of being the eye center
def PossibleCenter(Xcenter, Ycenter, coluna_final,linha_final,Xgradient,Ygradient):
		
		# initialize objective function
		objFunction = 0
		#n = 0
		
		# initialize x and y matrices
		indices = np.indices((linha_final,coluna_final))
		X = indices[1]
		Y = indices[0]
		
		# define the displacement vector matrices
		dx = X - Xcenter
		dy = Y - Ycenter
				
		# Normalize the displacement vector matrix
		magnitude = pow((dx*dx) + (dy*dy),0.5)
		dx = dx / (magnitude + 0.001)
		dy = dy / (magnitude + 0.001)
				
		# Normalize the luminous intensity gradient 
		gradMagnitude = pow((Xgradient**2) + (Ygradient**2),0.5)

		Xgradient = (Xgradient) / (gradMagnitude + 0.001)
				
		Ygradient = (Ygradient) / (gradMagnitude + 0.001)   
				
		# Evaluate dot product between gradient and displacement vectors
		dotProduct = dx*Xgradient + dy*Ygradient
		dotProduct = np.maximum(dotProduct,np.zeros((linha_final,coluna_final)))
		objFunction = np.sum(((255.0 - roi_new)**3)*(dotProduct**2))
		
		# Return objective function value normalized by the number of pixels squared 
		objFunction = (objFunction) / ((linha_final*coluna_final)**2)
		return objFunction

while (running == True):

	for event in pygame.event.get():                # verifying the quit event (click on x on page's top corner)
		if event.type == pygame.QUIT:           
			running = False                         # set running false and leaves the loop


	screen.fill(white)                              # Fill screen with white

	# Capture frame-by-frame
	ret, img = cap.read()
	
	# Apply Gaussian blurring
	blur = cv2.GaussianBlur(img,(5,5),0)
	
	# Change color-space from BGR to GrayScale
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	
	# Apply cascade face classifier
	eyes = eye_cascade.detectMultiScale(gray,1.3,9,
		0,(60, 60),(90, 90))
	eye_blink = eye_blink + 1  

	for (ex,ey,ew,eh) in eyes:
		
		if(1==1):
				
			roi_new = gray[ey:ey+eh, ex:ex+ew]
			
			cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
				
			# evaluate gradients via Sobel operators
			Xsobel = cv2.Sobel(roi_new,cv2.CV_64F,1,0,ksize=5)
			Ysobel = cv2.Sobel(roi_new,cv2.CV_64F,0,1,ksize=5)

			# Keep track of iterations
			n = 0

			# initialize X_0 and Y_0 for gradient "ascent"
			current_X =  ew/2
			current_Y =  eh/2

			########### Main Code ##########
			#------- Gradient "Ascent"  --------#       

			gamma = 1
			precision = 0.00002
			previous_step_size = 100
  
			while (abs(previous_step_size) > precision and n < 15):

				n = n + 1
	
				previous_X = current_X
				previous_Y = current_Y
				centralValue = PossibleCenter(previous_X,previous_Y,ew,eh,Xsobel,Ysobel)
				neighboor_X = PossibleCenter(previous_X+1,previous_Y,ew,eh,Xsobel,Ysobel)
				neighboor_Y = PossibleCenter(previous_X,previous_Y+1,ew,eh,Xsobel,Ysobel)
				gradX = neighboor_X - centralValue
				gradY = neighboor_Y - centralValue
	
				current_X = int(current_X + gamma*(gradX/abs(gradX)) )
				current_Y = int(current_Y + gamma*(gradY/abs(gradY)) )
	
				previous_step_size = (PossibleCenter(current_X,current_Y,ew,eh,Xsobel,Ysobel) 
				- centralValue)

				XcenterMax = current_X
				YcenterMax = current_Y
				delta_X = XcenterMax - ew/2
				delta_Y = YcenterMax - eh/2
				#print delta_X
				#print delta_Y
			cv2.line(img,(ex + XcenterMax - 4,ey + YcenterMax),
				(ex + XcenterMax + 4,ey + YcenterMax),(0,255,0),1)

			cv2.line(img,(ex + XcenterMax ,ey + YcenterMax - 4),
				(ex + XcenterMax ,ey + YcenterMax + 4),(0,255,0),1)

			if(delta_X > ew/9 and abs(delta_Y) < eh/14):
				right = right + 1

			if(delta_X < -ew/10 and abs(delta_Y) < eh/14):
				left = left + 1

			if(delta_Y > eh/20 and abs(delta_X) < ew/10):
				down = down + 1	

			if(delta_Y < -eh/18 and abs(delta_X) < ew/10):
				up = up + 1
					
			if(abs(delta_X) < 3 and abs(delta_Y) < 3):
				center = center + 1

			select = False
			if(eye_blink > 1):
				print 'XXXXxxxXXXX'
				select = True
			
			print eye_blink
			eye_blink = 0

			if(up > 3):
				sound1.play()
				print 'cima'
				up = 0
				if(Y_position > 240):
					Y_position = Y_position - 240
						
			if(down > 3):
				sound2.play()	
				print 'baixo'
				down = 0

				if(Y_position < 460):
					Y_position = Y_position + 240
				
					
			if(left > 5):
				sound4.play()
					
				print 'direita'
				left = 0  
				if(X_position < 1000):
					X_position = X_position + 240            	

			if(right > 3):
				sound3.play()
				if(X_position > 240):
					X_position = X_position - 240	
					
				print 'esquerda'
				right = 0

			if (center > 4):
				right = 0
				left = 0
				center = 0
				up = 0
				down = 0
				print 'center'
				
    # red selection rectangle
   	print "+++++"
   	print eyes
   	
   	if (len(eyes) > 0):

   		print "###############"
   		np.delete(eyes,0,0)
   		
   	if (len(eyes) > 1):
		np.delete(eyes,1,0)
	eyes = []
	print eyes

	pygame.draw.rect(screen,red,(X_position,Y_position,205,205), 5)

	if(X_position == 290 and Y_position == 20 and select == True):
		sound8.play()
		time.sleep(.600)

	if(X_position == 290 and Y_position == 260 and select == True):
		sound6.play()
		time.sleep(.600)

	if(X_position == 530 and Y_position == 20 and select == True):
		sound10.play()
		time.sleep(.600)

	if(X_position == 530 and Y_position == 260 and select == True):
		sound7.play()
		time.sleep(.600)

	if(X_position == 770 and Y_position == 20 and select == True):
		sound9.play()
		time.sleep(.600)

	if(X_position == 770 and Y_position == 260 and select == True):
		sound11.play()
		time.sleep(.600)

	# blit images on screen
	screen.blit(sono, (290,20))
	screen.blit(fome, (290,260))
	screen.blit(calor, (530,20))
	screen.blit(banheiro, (530,260))
	screen.blit(frio, (770,20))
	screen.blit(agua, (770,260))
    
 
	# Screen update, showing thing changing in time
	pygame.display.update()
	
	cv2.imshow('img',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break


# When dropping out the loop, quit pygame and close system
pygame.quit()  
sys.exit()

# Kill OpenCV windows
cap.release()
cv2.destroyAllWindows()

