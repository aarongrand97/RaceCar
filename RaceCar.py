import pygame, math, neat, sys, os
pygame.init()

# SET UP WINDOW
WIN_WIDTH = 467
WIN_HEIGHT = 700
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Racer")
trackImage = pygame.image.load('track.jpg')
trackImageScaled = pygame.transform.scale(trackImage, (WIN_WIDTH, WIN_HEIGHT))

RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
BLACK = (0,0,0)
WHITE = (255, 255, 255)

clock = pygame.time.Clock()

class Car:
    radius = 5
    current_angle = 90 # North up is 90, East is 0
    color = BLUE
    distances = [0, 0, 0, 0, 0] # distances to edge of track

    def __init__(self):
            self.xpos = 70
            self.ypos = 150

    def update(self):
            self.xpos += math.cos(self.current_angle*math.pi/180)
            self.ypos -= math.sin(self.current_angle*math.pi/180)

    def draw(self):
            pygame.draw.circle(win, self.color, (int(self.xpos), int(self.ypos)), self.radius)

    def checkIfGrassOrCurb(self, col):
        if(col == BLUE):
            return False
        elif (col[0] + col[1] + col[2] > 100): # any RGB that isnt roughly black
            return True
        return False

    def drawDistances(self): # draw lines from car to track edges
            angle_check = [self.current_angle, self.current_angle+45, self.current_angle+90, self.current_angle-45, self.current_angle-90]
            for ind, angle in enumerate(angle_check):
                    x , y = self.xpos + self.radius*math.cos(angle*math.pi/180) , self.ypos - self.radius*math.sin(angle*math.pi/180)
                    while(x < WIN_WIDTH and y < WIN_HEIGHT and self.checkIfGrassOrCurb(win.get_at((int(x),int(y)))) == False): # < 500 to ensure x,y do no exceed the window bounds causing crash, not ideal but works
                        x += 1 * math.cos(angle*math.pi/180)
                        y -= 1 * math.sin(angle*math.pi/180)
                        
                    self.distances[ind] = math.sqrt(((x-self.xpos)**2) + ((y-self.ypos)**2)) - self.radius # Store distances for use in NN
                        
                    x -= math.cos(angle*math.pi/180) # need to do this so the line doesnt cover up the black pixels
                    y += math.sin(angle*math.pi/180)
                    pygame.draw.line(win,BLUE, (self.xpos, self.ypos), (x,y), 2)

    def collide(self):
        col = win.get_at((int(self.xpos),int(self.ypos)))
        if(col[0] + col[1] + col[2] > 100 and col != BLUE): # this covers the grass and curb colours
            return True
        
        return False   


def draw(cars):
    win.blit(trackImageScaled, (0,0))
    for car in cars:
        car.draw()
        car.drawDistances()
    pygame.display.update()

def drawTrack():
    win.blit(trackImageScaled, (0,0))

def drawCars(cars):
    for car in cars:
        car.draw()
        car.drawDistances()

def main(genomes, config):
    # setup
    nets = []
    ge = []
    cars = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        cars.append(Car())
        g.fitness = 0
        ge.append(g)
    
    run = True
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
		pygame.quit()
	    # DEBUGGING FEATURE
	    if event.type == pygame.MOUSEBUTTONDOWN:
                print(len(cars))
                col = win.get_at(pygame.mouse.get_pos())
                print(col)

	drawTrack()
	
	for x, car in enumerate(cars):
            # get output from NN and apply
            output = nets[x].activate((car.distances[0], car.distances[1], car.distances[2], car.distances[3], car.distances[4]))
            if(output[0] > 0.5):
                car.current_angle += 2
                car.current_angle %= 360
            elif(output[0] < -0.5):
                car.current_angle -= 2
                car.current_angle %= 360
            car.update()
            ge[x].fitness += 0.1 # reward for surviving
            if (car.collide() == True):
                ge[x].fitness -= 5 # punish for crashing
                cars.pop(x) # remove from population
                nets.pop(x)
                ge.pop(x)
                
        if len(cars) < 1: # move onto next generation if all crashed
            run = False
            break
        drawCars(cars)
        pygame.display.update()
	
	clock.tick(60)

def run(config_path):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
	
	p = neat.Population(config)
	
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	
	winner = p.run(main,50)

				
if __name__ == "__main__":
        local_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(local_dir, "config_feedforward.txt")
        run(config_path)
