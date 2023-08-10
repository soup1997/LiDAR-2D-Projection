from collections import deque


dq = deque(maxlen=4)
dq.append(1)
dq.append(2)
dq.append(3)


print(dq[0]) # 3

dq.popleft()

print(dq[0])

print(dq)