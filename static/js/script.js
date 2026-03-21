(() => {
  function qs(sel) {
    return document.querySelector(sel);
  }

  function enable(el, yes) {
    if (!el) return;
    el.disabled = !yes;
  }

  function setHint(mode, hintEl) {
    if (!hintEl) return;
    if (mode === "search") {
      hintEl.textContent = "Search uses your keyword only (country/category not used).";
    } else {
      hintEl.textContent = "Top headlines uses country and category.";
    }
  }

  function showOverlay(show) {
    const overlay = qs("#loadingOverlay");
    if (!overlay) return;
    overlay.classList.toggle("d-none", !show);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const form = qs("#newsForm");
    const modeSelect = qs("#modeSelect");
    const keywordInput = qs("#keywordInput");
    const countrySelect = qs("#countrySelect");
    const categorySelect = qs("#categorySelect");
    const modeHint = qs("#modeHint");

    if (modeSelect) {
      modeSelect.addEventListener("change", () => {
        const isSearch = modeSelect.value === "search";
        enable(countrySelect, !isSearch);
        enable(categorySelect, !isSearch);
        setHint(modeSelect.value, modeHint);
      });
    }

    if (form) {
      form.addEventListener("submit", (e) => {
        const mode = modeSelect ? modeSelect.value : "top";
        const keyword = keywordInput ? (keywordInput.value || "").trim() : "";

        if (mode === "search" && !keyword) {
          e.preventDefault();
          alert("Please enter a keyword for Search mode.");
          keywordInput?.focus();
          return;
        }

        showOverlay(true);
      });
    }

    // Initial UI sync (in case server-side template didn't disable selects for some reason)
    const initialMode = modeSelect ? modeSelect.value : "top";
    const initialIsSearch = initialMode === "search";
    enable(countrySelect, !initialIsSearch);
    enable(categorySelect, !initialIsSearch);
    setHint(initialMode, modeHint);
  });
})();

