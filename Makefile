ML_final_summary.pdf: ML_final_summary.md
	pandoc ML_final_summary.md \
		--listings \
		--template=eisvogel \
		--pdf-engine=xelatex \
		-V mainfont="DejaVu Sans" \
		-o ML_final_summary.pdf

watch:
	echo ML_final_summary.md | entr make ML_final_summary.pdf

clean:
	rm ML_final_summary.pdf

.PHONY: clean
