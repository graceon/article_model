def str_to_num(s):
    # 初始化sign为 “+”，是因为开头是数字
    num, stack, sign = 0, [], '+'
    for i in range(len(s)):
        ch = s[i]
        if ch.isdigit():
            num = num * 10 + int(ch)
        # 根据当前数字之前的符号，来决定如何处理当前数值
        # 并将处理后的数值压入栈中
        if ch in "+-x/" or i == len(s) - 1:
            if sign == "+":
                stack.append(num)
            elif sign == "-":
                stack.append(-num)
            elif sign == "x":
                stack.append(stack.pop() * num)
            else:
                if num == 0:
                    break
                stack.append(int(stack.pop() / num))
            num = 0
            sign = ch
    return sum(stack)
