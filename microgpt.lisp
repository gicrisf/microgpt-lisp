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
                      v)))
    (print "running")))
