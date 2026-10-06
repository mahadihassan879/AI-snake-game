from turtle import Screen , Turtle
screen = Screen();
screen.title("snake game 2.0")
screen.bgcolor("#BC21D9")
screen.setup(width = 800 , height =800)
screen.tracer(0)


#create a snake body
segment = []
star_position =[ (0,0),(-20,0),(-40,0)]
for position in star_position :
    new_segment = Turtle("square") 
    new_segment.color = ("black")
    new_segment.penup()
    new_segment.shapesize(1,1)
    new_segment.goto(position)
    segment.append(new_segment)

screen.update()


screen.exitonclick()