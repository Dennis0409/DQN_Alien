import os
import random
import time
from enum import Enum
class obj(Enum):
    empty = 0
    wall = 1
    point = 2
    player = 3
    enemy = 4
    Exit = 5
class act(Enum):
    W = 0
    A = 1
    S = 2
    D = 3
class moveobj():
    def __init__(self,x,y) :
        self.x=x
        self.y=y

dir = {act.W:(-1,0),act.A:(0,-1),act.S:(1,0),act.D:(0,1)}
show = {obj.empty:" ",obj.enemy:"E",obj.player:"P",obj.point:"⋆",obj.wall:"1",obj.Exit:"D"}
humaninp={'W':act.W,'A':act.A,'S':act.S,'D':act.D}
class game:
    # 清屏函數
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    # 顯示遊戲畫面
    def display_screen(self):
        self.clear_screen()
        for i in range(self.width):
            for j in range(self.heigth):
                print(show[obj(self.map[i][j])],end=' ')
            print()
        print("Score:" , self.score)
    def __init__(self) :
        self.map=[
            [1,1,1],
            [2,0,4],
            [0,3,5]
                  ]
        self.player = moveobj(2,1)
        self.width = 3
        self.heigth = 3
        self.score = 0
        self.end = False
        self.display_screen()
        # initgame
    def get_state(self):
        return self.width
    def move(self,act):
        if(act in humaninp):
            act = humaninp[act]
        # 0W 1A 2S 3D
         
        newx=self.player.x
        newy=self.player.y
        if(act in dir):
            newx+=dir[act][0]
            newy+=dir[act][1]
            self.map[self.player.x][self.player.y]=obj.empty
            if(self.safe(newx,newy)):
                self.player.x=newx
                self.player.y=newy       
            self.player_move(newx,newy)
            
        self.display_screen()
        return self.score
    def player_move(self,x,y):
        if(self.map[x][y]==obj.point.value):
            self.score+=10
        elif(self.map[x][y]==obj.Exit.value):
            self.score+=10
            self.gameover()
            return
        elif(self.map[x][y]==obj.enemy.value):
            self.gameover()
            return
        self.map[self.player.x][self.player.y]=obj.player
    def gameover(self):
        print("gameover !")
        self.end=True
    def safe(self,x,y):
        if(not(x>=0 and x<=self.width and y>=0 and y<=self.heigth)):
            return False
        return  self.map[x][y]!=obj.wall.value
        

if __name__ == '__main__':
    gam=game()
    while(not gam.end):
        gam.move(input())
        
        