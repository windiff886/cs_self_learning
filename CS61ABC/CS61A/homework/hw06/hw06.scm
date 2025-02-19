(define (cddr s) (cdr (cdr s)))

(define (cadr s) 
  (car (cdr s)))

(define (caddr s) 
  (car (cdr (cdr s))))

(define (ascending? lst)
  (cond
    ((or (null? lst) (null? (cdr lst))) #t)
    ((> (car lst) (cadr lst)) #f)
    (else (ascending? (cdr lst)))))

(define (interleave lst1 lst2)
  (cond 
    ((null? lst1) lst2)     ; if lst1 is empty, return lst2
    ((null? lst2) lst1)     ; if lst2 is empty, return lst1
    (else (cons (car lst1)  ; else take first element from lst1
                (interleave lst2 (cdr lst1))))))

(define (my-filter func lst)
  (cond 
    ((null? lst) nil)                              ; 如果列表为空，返回空列表
    ((func (car lst))                              ; 如果当前元素满足谓词函数
     (cons (car lst) (my-filter func (cdr lst))))  ; 保留该元素并继续处理剩余部分
    (else (my-filter func (cdr lst)))))            ; 否则跳过当前元素继续处理

(define (no-repeats lst)
  (if (null? lst)
      nil
      (cons (car lst)
            (no-repeats 
             (my-filter (lambda (x) (not (= x (car lst))))
                       (cdr lst))))))
