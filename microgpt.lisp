;;;; microgpt.lisp

(in-package #:microgpt)

(defun load-names ()
  (let ((path (asdf:system-relative-pathname :microgpt "input.txt"))
        (names (make-array 0 :element-type 'string
                             :adjustable t
                             :fill-pointer 0)))
    (with-open-file (stream path :direction :input)
      (loop for line = (read-line stream nil nil)
            while line
            do (vector-push-extend line names)))
    names))

(defun shuffle (vec)
  "Shuffle vector VEC in place using Fisher-Yates."
  (loop for i from (1- (length vec)) downto 1
        do (rotatef (aref vec i) (aref vec (random (1+ i)))))
  vec)

(defun unique-chars (names)
  "Return a vector of unique characters found across all strings in NAMES."
  (let ((seen (make-hash-table)))
    (loop for name across names
          do (loop for ch across name
                   do (setf (gethash ch seen) t)))
    (loop for ch being the hash-keys of seen
          collect ch into chars
          finally (return (coerce chars 'vector)))))

;; Box-Muller transform for standard normal (mean 0, std dev 0.08)
(defun random-gauss (&optional (mean 0.0) (std-dev 0.8))
  (let ((u1 (random 1.0))
        (u2 (random 1.0)))
    (+ mean
       (* std-dev
          (sqrt (* -2.0 (log u1)))
          (cos (* 2.0 pi u2))))))

(defun matrix (nout nin)
  "Create a matrix of value objects with random gaussian init.

Args:
  NOUT (integer): Number of output rows.
  NIN (integer): Number of input columns."
  (loop for _i from 0 below nout
        collect (loop for _j from 0 below nin
                      collect (random-gauss))))

(defun main ()
  (let* ((names (let ((n (shuffle (load-names))))
                  (format t "Names: ~a~%" (length n))
                  n))
         ;; Let there be a Tokenizer
         ;; to translate strings to discrete symbols and back
         ;;
         ;; unique characters in the dataset become token ids 0..n-1
         (uchars (sort (unique-chars names) #'char<))
         ;; token id for the special Beginning of Sequence (BOS) token
         (bos (length uchars))
         ;; total number of unique tokens, +1 is for BOS
         (vocab-size (let ((v (+ bos 1)))
                      (format t "Vocab size: ~a~%" v)
                      v))
         (n-embd 16)  ;; embedding dimension
         (n-head 4)   ;; number of attention heads
         (n-layer 1)  ;; number of layers
         (block-size 16) ;; maximum sequence length
         (head-dim (/ n-embd n-head)) ;; dimension of each head
         ;; TODO Implement value
         (state-table (let ((ht (make-hash-table :test 'equal)))
                        (setf (gethash "wte" ht) (matrix vocab-size n-embd)
                              (gethash "wpe" ht) (matrix block-size n-embd)
                              (gethash "lm_head" ht) (matrix vocab-size n-embd))
                        (loop for i from 0 below n-layer
                              do (setf (gethash (format nil "layer~a.attn_wq" i) ht) (matrix n-embd n-embd)
                                       (gethash (format nil "layer~a.attn_wk" i) ht) (matrix n-embd n-embd)
                                       (gethash (format nil "layer~a.attn_wv" i) ht) (matrix n-embd n-embd)
                                       (gethash (format nil "layer~a.attn_wo" i) ht) (matrix n-embd n-embd)
                                       (gethash (format nil "layer~a.mlp_fc1" i) ht) (matrix (* 4 n-embd) n-embd)
                                       (gethash (format nil "layer~a.mlp_fc2" i) ht) (matrix n-embd (* 4 n-embd))))
                        ht)))
    ;; Debug print of the state table
    (maphash (lambda (k v)
               (format t "~a (~a x ~a):~%" k (length v) (length (first v)))
               (loop for row in (subseq v 0 (min 3 (length v)))
                     do (format t "  ~{~,4f ~}~%" (subseq row 0 (min 5 (length row)))))
               (format t "  ...~%"))
             state-table)))
