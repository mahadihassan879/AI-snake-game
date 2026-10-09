import time
import random
from turtle import Screen, Turtle

# ----------------------------------------
# 1. SETUP THE SCREEN
# ----------------------------------------
screen = Screen()
screen.title("Snake Game 2.0")
screen.bgcolor("#BC21D9")  # Your purple background!
screen.setup(width=600, height=600)
screen.tracer(0) 

# ----------------------------------------
# 2. CREATE THE SNAKE
# ----------------------------------------
segments = []
start_positions = [(0, 0), (-20, 0), (-40, 0)]

def add_segment(position):
    new_segment = Turtle("square")
    new_segment.color("black")
    new_segment.penup()
    new_segment.goto(position)
    segments.append(new_segment)

# Build the starting snake
for position in start_positions:
    add_segment(position)

head = segments[0]
head.direction = "Right" # The snake starts by moving right

# ----------------------------------------
# 3. CREATE THE FOOD
# ----------------------------------------
food = Turtle("circle")
food.color("yellow")
food.penup()
food.goto(100, 100)

# ----------------------------------------
# 4. MOVEMENT CONTROLS
# ----------------------------------------
# These prevent the snake from instantly reversing into itself
def go_up():
    if head.direction != "Down":
        head.direction = "Up"

def go_down():
    if head.direction != "Up":
        head.direction = "Down"

def go_left():
    if head.direction != "Right":
        head.direction = "Left"

def go_right():
    if head.direction != "Left":
        head.direction = "Right"

# Tell the screen to listen for keyboard presses
screen.listen()
screen.onkey(go_up, "Up")
screen.onkey(go_down, "Down")
screen.onkey(go_left, "Left")
screen.onkey(go_right, "Right")

# ----------------------------------------
# 5. MAIN GAME LOOP
# ----------------------------------------
game_is_on = True

while game_is_on:
    screen.update()
    time.sleep(0.1)

    # A. Check if the snake eats the food
    if head.distance(food) < 20:
        # Move food to a random new spot
        new_x = random.randint(-270, 270)
        new_y = random.randint(-270, 270)
        food.goto(new_x, new_y)
        
        # Add a new block to the very end of the snake
        add_segment(segments[-1].position())

    # B. Move the snake's body (The "Conga Line")
    for seg_num in range(len(segments)-1, 0, -1):
        new_x = segments[seg_num - 1].xcor()
        new_y = segments[seg_num - 1].ycor()
        segments[seg_num].goto(new_x, new_y)

    # C. Move the snake's head based on the current direction
    if head.direction == "Up":
        head.sety(head.ycor() + 20)
    elif head.direction == "Down":
        head.sety(head.ycor() - 20)
    elif head.direction == "Left":
        head.setx(head.xcor() - 20)
    elif head.direction == "Right":
        head.setx(head.xcor() + 20)

    # D. Check for Game Over (Hitting the wall)
    if head.xcor() > 290 or head.xcor() < -290 or head.ycor() > 290 or head.ycor() < -290:
        game_is_on = False
        head.color("red") # Turn the head red when it crashes

    # E. Check for Game Over (Hitting its own tail)
    for segment in segments[1:]:
        if head.distance(segment) < 10:
            game_is_on = False
            head.color("red")

# Keeps the screen open until you click on it after a game over
screen.exitonclick()