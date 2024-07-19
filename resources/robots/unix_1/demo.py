import pybullet as p 
import time 
import pybullet_data



import pybullet as p
import time

# class Dog:
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version 
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally 

planeId = p.loadURDF("plane.urdf") 

p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)

urdfFlags = p.URDF_USE_SELF_COLLISION
# 位置 姿态 urdfFlags 固定基座
quadruped = p.loadURDF("urdf/unix_1.urdf", [0, 0, 1.6], [0, 0, 0, 1], flags = urdfFlags, useFixedBase=True)
# quadruped = p.loadURDF("urdf/cassie.urdf", [0, 0, 1], [0, 0, 0, 1], flags = urdfFlags, useFixedBase=True)

# 获取机器人关节数量
num_joints = p.getNumJoints(quadruped)
print(f"机器人关节数量: {num_joints}")

# 打印每个关节的信息以调试
for j in range (p.getNumJoints(quadruped)):
        print(p.getJointInfo(quadruped,j))

#enable collision between lower legs
left_legs = [3,4,5]
right_legs = [8,9,10]
for l0 in left_legs:
    for l1 in right_legs:
        if (l1>l0):
            enableCollision = 1
            try:
                bodyA = p.getJointInfo(quadruped, l0)[16]  # 子刚体索引
                bodyB = p.getJointInfo(quadruped, l1)[16]  # 子刚体索引
                print("collision for pair", l0, l1, p.getJointInfo(quadruped, l0)[12], p.getJointInfo(quadruped, l1)[12], "enabled=", enableCollision)
                p.setCollisionFilterPair(quadruped, quadruped, bodyA, bodyB, enableCollision)
            except Exception as e:
                print(f"设置碰撞过滤器对失败: {e}")
            # print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
            # p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

jointIds=[]
paramIds=[]

maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

for j in range (p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped,j)
    print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        jointIds.append(j)

# print(jointIds)

p.getCameraImage(480,320)
p.setRealTimeSimulation(0)

joints=[]

# p.disconnect()
for j in range (p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped,j)
    js = p.getJointState(quadruped,j)
    #print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,(js[0]-0)/1))


p.setRealTimeSimulation(1)

while (1):
    for i in range(len(paramIds)):
        c = paramIds[i]
        targetPos = p.readUserDebugParameter(c)
        maxForce = p.readUserDebugParameter(maxForceId)
        p.setJointMotorControl2(quadruped,jointIds[i],p.POSITION_CONTROL,1*targetPos+0, force=maxForce)
