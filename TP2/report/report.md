# Report
Potier - Boireau

## Exercise 1 - Hough transform for circles
1. Let $ r \in [1, 100] $ 
- If $\delta r = 2 $, we have 50 discrete values
- If $\delta r = 0.5 $, we have 200 discrete values

2. Let $ r, c \in [1, 100] $, $ rad \in [5, 100\sqrt2] $ and $\delta r$ = $\delta c$ = $\delta rad$ = 1
- For r and c we have 100 discrete values each
- For rad we have $\frac{rad_{max} - rad_{min}}{\delta_{rad}}$ = $\frac{100\sqrt2 - 5}{1}$ = 137 discrete values
> So we can describe $ 100 \times 100 \times 137$ = $1 370 000$ circles

3.
In the general case we have $acc(r, c, rad)$ corresponding to the circle with:
- center = $(r_{min} + (i-1) \times \delta r, c_{min} + (j-1) \times \delta c)$ 
- rad = $rad_{min} + (k-1) \times \delta rad$

If we take the intervals and steps from the last question:
- $acc(1, 1, 1)$ corresponds to the circle with:
    - center = $(1, 1)$ 
    - rad = $5$
- $acc(10, 7, 30)$ corresponds to the circle with:
    - center = $(1 + (10-1), 1 + (7-1))$ = $(10, 7)$
    - rad = $5 + (30-1)$ = $34$