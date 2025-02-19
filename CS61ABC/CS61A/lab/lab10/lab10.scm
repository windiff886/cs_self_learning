(define (over-or-under num1 num2) 
    'YOUR-CODE-HERE
    (cond 
        ((< num1 num2) -1)
        ((= num1 num2) 0)
        ((> num1 num2) 1)
    )
)

(define (make-adder num) 
    (lambda (inc) (+ inc num))
)

(define (composed f g)
  (lambda (x)
    (f (g x))))

(define (square n) (* n n))

(define (pow base exp)
  (if (= exp 0)
      1
      (let ((half (pow base (quotient exp 2))))
        (if (= (remainder exp 2) 0)
            (* half half)
            (* base half half)))))
