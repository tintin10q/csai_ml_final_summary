ML_final_summary.pdf: ML_final_summary.md
	pandoc ML_final_summary.md \
		--listings \
		--template=eisvogel \
		-o ML_final_summary.pdf

watch:
	echo ML_final_summary.md | entr make report.pdf

clean:
	rm ML_final_summary.md.pdf

.PHONY: clean
