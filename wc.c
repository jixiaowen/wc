#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <stdint.h>

#define BUFFER_SIZE 65536

typedef struct {
    uint64_t lines;
    uint64_t words;
    uint64_t bytes;
    uint64_t chars;
} Count;

typedef struct {
    int show_lines;
    int show_words;
    int show_bytes;
    int show_chars;
    unsigned char *word_delim;
    size_t word_delim_len;
    unsigned char *line_delim;
    size_t line_delim_len;
} Options;

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s [-l] [-w] [-c] [-m] [-d DELIM] [-L LINEDELIM] [FILE...]\n", prog);
}

static void count_whitespace_words(const unsigned char *buf, size_t n, int *in_word, uint64_t *words) {
    for (size_t i = 0; i < n; i++) {
        if (isspace(buf[i])) {
            *in_word = 0;
        } else if (!*in_word) {
            (*words)++;
            *in_word = 1;
        }
    }
}

static void count_delimiter_words(const unsigned char *buf, size_t n, const unsigned char *delim, size_t delim_len, int *in_token, uint64_t *words, int finalize, unsigned char *carry, size_t *carry_len) {
    size_t work_len = *carry_len + n;
    if (work_len == 0) {
        *carry_len = 0;
        return;
    }
    unsigned char *work = malloc(work_len);
    if (!work) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }
    if (*carry_len > 0) {
        memcpy(work, carry, *carry_len);
    }
    if (n > 0) {
        memcpy(work + *carry_len, buf, n);
    }

    size_t limit;
    if (finalize) {
        limit = work_len;
    } else {
        if (work_len < delim_len) {
            limit = 0;
        } else {
            limit = work_len - delim_len + 1;
        }
    }

    size_t i = 0;
    int has_full = work_len >= delim_len;
    size_t max_start = has_full ? work_len - delim_len : 0;

    while (i < limit) {
        int is_delim = 0;
        if (has_full && i <= max_start) {
            if (memcmp(work + i, delim, delim_len) == 0) {
                is_delim = 1;
            }
        }
        if (is_delim) {
            if (*in_token) {
                (*words)++;
                *in_token = 0;
            }
            i += delim_len;
        } else {
            *in_token = 1;
            i++;
        }
    }

    if (!finalize && delim_len > 1) {
        size_t new_carry_len = work_len - limit;
        if (new_carry_len > 0) {
            memmove(carry, work + limit, new_carry_len);
        }
        *carry_len = new_carry_len;
    } else {
        *carry_len = 0;
    }

    free(work);
}

static void count_delimiter_occurrences(const unsigned char *buf, size_t n, const unsigned char *delim, size_t delim_len, uint64_t *count, int finalize, unsigned char *carry, size_t *carry_len) {
    size_t work_len = *carry_len + n;
    if (work_len == 0) {
        *carry_len = 0;
        return;
    }
    unsigned char *work = malloc(work_len);
    if (!work) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }
    if (*carry_len > 0) {
        memcpy(work, carry, *carry_len);
    }
    if (n > 0) {
        memcpy(work + *carry_len, buf, n);
    }

    size_t limit;
    if (finalize) {
        limit = work_len;
    } else {
        if (work_len < delim_len) {
            limit = 0;
        } else {
            limit = work_len - delim_len + 1;
        }
    }

    size_t i = 0;
    int has_full = work_len >= delim_len;
    size_t max_start = has_full ? work_len - delim_len : 0;

    while (i < limit) {
        int is_delim = 0;
        if (has_full && i <= max_start) {
            if (memcmp(work + i, delim, delim_len) == 0) {
                is_delim = 1;
            }
        }
        if (is_delim) {
            (*count)++;
            i += delim_len;
        } else {
            i++;
        }
    }

    if (!finalize && delim_len > 1) {
        size_t new_carry_len = work_len - limit;
        if (new_carry_len > 0) {
            memmove(carry, work + limit, new_carry_len);
        }
        *carry_len = new_carry_len;
    } else {
        *carry_len = 0;
    }

    free(work);
}

static int hex_value(int ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    }
    return -1;
}

static int parse_escaped(const char *in, unsigned char **out, size_t *out_len) {
    size_t in_len = strlen(in);
    unsigned char *buf = malloc(in_len + 1);
    if (!buf) {
        fprintf(stderr, "memory allocation failed\n");
        return -1;
    }
    size_t j = 0;
    for (size_t i = 0; i < in_len; i++) {
        unsigned char ch = (unsigned char)in[i];
        if (ch == '\\' && i + 1 < in_len) {
            unsigned char next = (unsigned char)in[++i];
            if (next == 'n') {
                buf[j++] = '\n';
            } else if (next == 'r') {
                buf[j++] = '\r';
            } else if (next == 't') {
                buf[j++] = '\t';
            } else if (next == '\\') {
                buf[j++] = '\\';
            } else if (next == '0') {
                buf[j++] = '\0';
            } else if (next == 'x' && i + 2 < in_len) {
                int hi = hex_value((unsigned char)in[i + 1]);
                int lo = hex_value((unsigned char)in[i + 2]);
                if (hi >= 0 && lo >= 0) {
                    buf[j++] = (unsigned char)((hi << 4) | lo);
                    i += 2;
                } else {
                    buf[j++] = next;
                }
            } else {
                buf[j++] = next;
            }
        } else {
            buf[j++] = ch;
        }
    }
    *out = buf;
    *out_len = j;
    return 0;
}

static int count_stream(FILE *fp, const Options *opt, Count *out) {
    unsigned char buf[BUFFER_SIZE];
    size_t n;
    int in_word = 0;
    int in_token = 0;
    size_t word_delim_len = opt->word_delim_len;
    size_t line_delim_len = opt->line_delim_len;
    unsigned char *word_carry = NULL;
    size_t word_carry_len = 0;
    unsigned char *line_carry = NULL;
    size_t line_carry_len = 0;

    if (word_delim_len > 0) {
        word_carry = malloc(word_delim_len);
        if (!word_carry) {
            fprintf(stderr, "memory allocation failed\n");
            return -1;
        }
    }
    if (line_delim_len > 0) {
        line_carry = malloc(line_delim_len);
        if (!line_carry) {
            fprintf(stderr, "memory allocation failed\n");
            free(word_carry);
            return -1;
        }
    }

    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        out->bytes += n;
        if (opt->show_chars) {
            out->chars += n;
        }
        if (opt->show_lines) {
            if (line_delim_len == 0) {
                for (size_t i = 0; i < n; i++) {
                    if (buf[i] == '\n') {
                        out->lines++;
                    }
                }
            } else if (line_delim_len == 1) {
                unsigned char d = opt->line_delim[0];
                for (size_t i = 0; i < n; i++) {
                    if (buf[i] == d) {
                        out->lines++;
                    }
                }
            } else {
                count_delimiter_occurrences(buf, n, opt->line_delim, line_delim_len, &out->lines, 0, line_carry, &line_carry_len);
            }
        }

        if (opt->show_words) {
            if (word_delim_len == 0) {
                count_whitespace_words(buf, n, &in_word, &out->words);
            } else {
                count_delimiter_words(buf, n, opt->word_delim, word_delim_len, &in_token, &out->words, 0, word_carry, &word_carry_len);
            }
        }
    }

    if (opt->show_lines && line_delim_len > 1) {
        count_delimiter_occurrences(NULL, 0, opt->line_delim, line_delim_len, &out->lines, 1, line_carry, &line_carry_len);
    }

    if (opt->show_words && word_delim_len > 0) {
        count_delimiter_words(NULL, 0, opt->word_delim, word_delim_len, &in_token, &out->words, 1, word_carry, &word_carry_len);
        if (in_token) {
            out->words++;
        }
    }

    free(word_carry);
    free(line_carry);

    if (ferror(fp)) {
        return -1;
    }
    return 0;
}

static void print_counts(const Count *c, const Options *opt, const char *name) {
    if (opt->show_lines) {
        printf("%7llu", (unsigned long long)c->lines);
    }
    if (opt->show_words) {
        printf("%7llu", (unsigned long long)c->words);
    }
    if (opt->show_bytes) {
        printf("%7llu", (unsigned long long)c->bytes);
    }
    if (opt->show_chars) {
        printf("%7llu", (unsigned long long)c->chars);
    }
    if (name) {
        printf(" %s", name);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    Options opt = {0, 0, 0, 0, NULL, 0, NULL, 0};
    static struct option long_opts[] = {
        {"lines", no_argument, 0, 'l'},
        {"words", no_argument, 0, 'w'},
        {"bytes", no_argument, 0, 'c'},
        {"chars", no_argument, 0, 'm'},
        {"delimiter", required_argument, 0, 'd'},
        {"line-delimiter", required_argument, 0, 'L'},
        {0, 0, 0, 0}
    };

    int ch;
    while ((ch = getopt_long(argc, argv, "lwcmd:L:", long_opts, NULL)) != -1) {
        switch (ch) {
            case 'l':
                opt.show_lines = 1;
                break;
            case 'w':
                opt.show_words = 1;
                break;
            case 'c':
                opt.show_bytes = 1;
                break;
            case 'm':
                opt.show_chars = 1;
                break;
            case 'd':
                free(opt.word_delim);
                opt.word_delim = NULL;
                opt.word_delim_len = 0;
                if (parse_escaped(optarg, &opt.word_delim, &opt.word_delim_len) != 0) {
                    return 1;
                }
                break;
            case 'L':
                free(opt.line_delim);
                opt.line_delim = NULL;
                opt.line_delim_len = 0;
                if (parse_escaped(optarg, &opt.line_delim, &opt.line_delim_len) != 0) {
                    return 1;
                }
                break;
            default:
                usage(argv[0]);
                return 2;
        }
    }

    if (!opt.show_lines && !opt.show_words && !opt.show_bytes && !opt.show_chars) {
        opt.show_lines = 1;
        opt.show_words = 1;
        opt.show_bytes = 1;
    }

    int file_count = argc - optind;
    int had_error = 0;
    Count total = {0, 0, 0, 0};

    if (file_count == 0) {
        Count c = {0, 0, 0, 0};
        if (count_stream(stdin, &opt, &c) != 0) {
            fprintf(stderr, "read error\n");
            return 1;
        }
        print_counts(&c, &opt, NULL);
        free(opt.word_delim);
        free(opt.line_delim);
        return 0;
    }

    for (int i = optind; i < argc; i++) {
        const char *name = argv[i];
        FILE *fp;
        if (strcmp(name, "-") == 0) {
            fp = stdin;
        } else {
            fp = fopen(name, "rb");
        }
        if (!fp) {
            fprintf(stderr, "%s: %s\n", name, strerror(errno));
            had_error = 1;
            continue;
        }
        Count c = {0, 0, 0, 0};
        if (count_stream(fp, &opt, &c) != 0) {
            fprintf(stderr, "%s: read error\n", name);
            had_error = 1;
        }
        if (fp != stdin) {
            fclose(fp);
        }
        print_counts(&c, &opt, name);
        total.lines += c.lines;
        total.words += c.words;
        total.bytes += c.bytes;
        total.chars += c.chars;
    }

    if (file_count > 1) {
        print_counts(&total, &opt, "total");
    }

    free(opt.word_delim);
    free(opt.line_delim);
    return had_error ? 1 : 0;
}
