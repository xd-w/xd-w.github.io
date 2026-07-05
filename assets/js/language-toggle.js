(function() {
  var languageStorageKey = "xudong-site-language-v3";
  var themeStorageKey = "xudong-site-theme-v3";
  var hashPairs = {
    "#about": "#about-zh",
    "#research": "#research-zh",
    "#latest-news": "#latest-news-zh",
    "#about-zh": "#about",
    "#research-zh": "#research",
    "#latest-news-zh": "#latest-news"
  };

  function normalizeLanguage(lang) {
    return lang === "zh" ? "zh" : "en";
  }

  function normalizeTheme(theme) {
    return theme === "dark" ? "dark" : "light";
  }

  function getStoredLanguage() {
    try {
      return normalizeLanguage(window.localStorage.getItem(languageStorageKey));
    } catch (error) {
      return "en";
    }
  }

  function getStoredTheme() {
    try {
      var storedTheme = window.localStorage.getItem(themeStorageKey);
      return storedTheme === "dark" || storedTheme === "light" ? storedTheme : "";
    } catch (error) {
      return "";
    }
  }

  function getSystemTheme() {
    try {
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return "dark";
      }
    } catch (error) {
      return "light";
    }
    return "light";
  }

  function getPreferredTheme() {
    var storedTheme = getStoredTheme();
    if (storedTheme) {
      return storedTheme;
    }
    return getSystemTheme();
  }

  function storeLanguage(lang) {
    try {
      window.localStorage.setItem(languageStorageKey, lang);
    } catch (error) {
      return;
    }
  }

  function storeTheme(theme) {
    try {
      window.localStorage.setItem(themeStorageKey, theme);
    } catch (error) {
      return;
    }
  }

  function setNavTargets(lang) {
    var links = document.querySelectorAll("[data-href-en][data-href-zh]");
    Array.prototype.forEach.call(links, function(link) {
      link.setAttribute("href", lang === "zh" ? link.getAttribute("data-href-zh") : link.getAttribute("data-href-en"));
    });
  }

  function updateLanguageButtons(lang) {
    var buttons = document.querySelectorAll("[data-lang-toggle]");
    Array.prototype.forEach.call(buttons, function(button) {
      button.setAttribute("aria-pressed", lang === "zh" ? "true" : "false");
      button.setAttribute("aria-label", lang === "zh" ? "Switch to English" : "切换到中文");
      button.setAttribute("title", lang === "zh" ? "Switch to English" : "切换到中文");
    });
  }

  function updateThemeButtons(theme) {
    var buttons = document.querySelectorAll("[data-theme-toggle]");
    Array.prototype.forEach.call(buttons, function(button) {
      var nextTheme = theme === "dark" ? "light" : "dark";
      button.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
      button.setAttribute("aria-label", nextTheme === "dark" ? "Switch to dark theme" : "Switch to light theme");
      button.setAttribute("title", nextTheme === "dark" ? "Switch to dark theme" : "Switch to light theme");
    });
  }

  function updateThemeColor(theme) {
    var meta = document.querySelector("meta[name='theme-color']");
    if (meta) {
      meta.setAttribute("content", theme === "dark" ? "#111315" : "#ffffff");
    }
  }

  function setLanguage(lang, shouldStore) {
    lang = normalizeLanguage(lang);
    document.documentElement.setAttribute("data-lang", lang);
    document.documentElement.setAttribute("lang", lang === "zh" ? "zh-Hans" : "en");
    setNavTargets(lang);
    updateLanguageButtons(lang);

    if (shouldStore) {
      storeLanguage(lang);
    }
  }

  function setTheme(theme, shouldStore) {
    theme = normalizeTheme(theme);
    document.documentElement.setAttribute("data-theme", theme);
    updateThemeButtons(theme);
    updateThemeColor(theme);

    if (shouldStore) {
      storeTheme(theme);
    }
  }

  function syncHashForLanguage(nextLang) {
    var currentHash = window.location.hash;
    if (!currentHash || !hashPairs[currentHash]) {
      return null;
    }

    if (hashMatchesLanguage(currentHash, nextLang)) {
      return null;
    }

    return hashPairs[currentHash];
  }

  function hashMatchesLanguage(hash, lang) {
    var isChineseHash = hash.slice(-3) === "-zh";
    return (lang === "zh" && isChineseHash) || (lang !== "zh" && !isChineseHash);
  }

  function getVisibleHash(hash, lang) {
    if (!hash || !hashPairs[hash] || hashMatchesLanguage(hash, lang)) {
      return hash;
    }

    return hashPairs[hash];
  }

  function getSamePageHash(link) {
    var href = link.getAttribute("href");
    if (!href) {
      return "";
    }

    try {
      var url = new URL(href, window.location.href);
      if (url.origin !== window.location.origin || url.pathname !== window.location.pathname) {
        return "";
      }
      return url.hash;
    } catch (error) {
      return href.charAt(0) === "#" ? href : "";
    }
  }

  function expandTarget(target) {
    if (!target) {
      return;
    }

    var details = target.matches("details") ? target : target.querySelector("details");
    if (!details && target.closest) {
      details = target.closest("details");
    }

    if (details) {
      details.open = true;
    }
  }

  function expandHashTarget(shouldScroll) {
    if (!window.location.hash) {
      return;
    }

    var lang = normalizeLanguage(document.documentElement.getAttribute("data-lang"));
    var hash = getVisibleHash(window.location.hash, lang);
    var target = hash ? document.querySelector(hash) : null;
    if (!target) {
      return;
    }

    if (hash !== window.location.hash) {
      window.history.replaceState(null, "", hash);
    }

    expandTarget(target);

    if (shouldScroll) {
      window.requestAnimationFrame(function() {
        var headerOffset = 72;
        var targetY = target.getBoundingClientRect().top + window.pageYOffset - headerOffset;
        window.scrollTo({ top: Math.max(0, targetY), behavior: "smooth" });
      });
    }
  }

  function scrollToHash(hash, shouldReplace) {
    if (!hash) {
      return false;
    }

    var target = document.querySelector(hash);
    if (!target) {
      return false;
    }

    expandTarget(target);

    if (shouldReplace) {
      window.history.replaceState(null, "", hash);
    } else {
      window.history.pushState(null, "", hash);
    }

    window.requestAnimationFrame(function() {
      var headerOffset = 72;
      var targetY = target.getBoundingClientRect().top + window.pageYOffset - headerOffset;
      window.scrollTo({ top: Math.max(0, targetY), behavior: "smooth" });
    });

    return true;
  }

  function bindSystemThemeSync() {
    var query = null;
    try {
      if (!window.matchMedia) {
        return;
      }
      query = window.matchMedia("(prefers-color-scheme: dark)");
    } catch (error) {
      return;
    }

    var handleSystemThemeChange = function(event) {
      if (getStoredTheme()) {
        return;
      }
      setTheme(event.matches ? "dark" : "light", false);
    };

    if (query.addEventListener) {
      query.addEventListener("change", handleSystemThemeChange);
    } else if (query.addListener) {
      query.addListener(handleSystemThemeChange);
    }
  }

  document.addEventListener("DOMContentLoaded", function() {
    var currentLang = normalizeLanguage(document.documentElement.getAttribute("data-lang") || getStoredLanguage());
    var currentTheme = normalizeTheme(document.documentElement.getAttribute("data-theme") || getPreferredTheme());
    setLanguage(currentLang, false);
    setTheme(currentTheme, false);
    bindSystemThemeSync();
    expandHashTarget(true);

    document.addEventListener("click", function(event) {
      var link = event.target.closest && event.target.closest("a[href]");
      if (!link || !link.matches(".nav-link-i18n, .profile-hero__actions a")) {
        return;
      }

      var hash = getSamePageHash(link);
      if (!hash || !document.querySelector(hash)) {
        return;
      }

      event.preventDefault();
      event.stopImmediatePropagation();
      scrollToHash(hash, false);
    }, true);

    var languageButtons = document.querySelectorAll("[data-lang-toggle]");
    Array.prototype.forEach.call(languageButtons, function(button) {
      button.addEventListener("click", function() {
        var nextLang = document.documentElement.getAttribute("data-lang") === "zh" ? "en" : "zh";
        setLanguage(nextLang, true);
        var nextHash = syncHashForLanguage(nextLang);
        if (nextHash) {
          scrollToHash(nextHash, true);
        } else {
          expandHashTarget(false);
        }
      });
    });

    var themeButtons = document.querySelectorAll("[data-theme-toggle]");
    Array.prototype.forEach.call(themeButtons, function(button) {
      button.addEventListener("click", function() {
        var nextTheme = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        setTheme(nextTheme, true);
      });
    });

    window.addEventListener("hashchange", function() {
      expandHashTarget(true);
    });
  });
})();
